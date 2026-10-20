"""Closed-loop replay of an agent workload (agent_trace.py) against an OpenAI-compatible server.

Each session is a chain: wait (real tool latency / human think time) -> send round -> stream the whole response ->
next round. The prompt is the session's running context, so the server's prefix cache sees what a coding agent
produces: context = shared system block + the session's history; each round appends its (scaled) new tokens, and the
previous output is re-sent as input (Claude's "output-resend" pattern, TraceLab Sec. 5.3). Token ids are synthetic
(the trace has no text): the shape (lengths, waits, order, arrivals) is the trace's. Output length is forced
(min_tokens = max_tokens, ignore_eos). A context that would pass --max-ctx is compacted to the system block +
--compact-to fresh tokens (what Claude Code's auto-compaction does).

One JSONL record per round: session, task, kind, t_send, ttft, tbt_mean/p99/max, e2e, prompt/completion tokens.

  python agent_replay.py --url http://127.0.0.1:8000/v1 --model llama3 --workload agent_workload.json --out rounds.jsonl
"""
import argparse
import asyncio
import json
import random
import time

from openai import AsyncOpenAI

SYS_LEN = 2048  # shared across sessions: the agent's system prompt + tool schemas (~16K tokens in Claude Code, x1/8)


def ids(rng, n):
    return [rng.randrange(1000, 150000) for _ in range(n)]


async def session(client, a, s, t0, out, lock, sys_block):
    rng = random.Random(1000 + s["id"])
    await asyncio.sleep(max(0.0, t0 + s["arrival"] - time.time()))
    ctx = sys_block + ids(rng, max(0, s["prefix0"] - SYS_LEN))
    task = -1
    for k, r in enumerate(s["rounds"]):
        await asyncio.sleep(r["wait"])
        if r["kind"] == "user":
            task += 1
        compacted = len(ctx) + r["append"] + r["output"] > a.max_ctx
        if compacted:
            ctx = sys_block + ids(rng, a.compact_to)
        ctx = ctx + ids(rng, min(r["append"], a.max_ctx - len(ctx) - r["output"]))
        ts = time.time()
        ttft, chunks, usage, err = None, [], None, None
        try:
            stream = await client.completions.create(
                model=a.model, prompt=ctx, max_tokens=r["output"], temperature=0.0, stream=True,
                stream_options={"include_usage": True},
                extra_body={"ignore_eos": True, "min_tokens": r["output"]})
            async for ch in stream:
                if ch.usage:
                    usage = ch.usage
                if ch.choices and ch.choices[0].text:
                    now = time.time()
                    ttft = ttft if ttft is not None else now - ts
                    chunks.append(now)
        except Exception as e:  # keep the session going; the record says what failed
            err = repr(e)
        te = time.time()
        gaps = sorted(b - c for c, b in zip(chunks, chunks[1:]))
        rec = {"session": s["id"], "round": k, "task": task, "kind": r["kind"], "wait": r["wait"],
               "t_send": ts, "t_done": te, "ttft": ttft, "e2e": te - ts, "compacted": compacted,
               "tbt_mean": sum(gaps) / len(gaps) if gaps else None,
               "tbt_p99": gaps[int(0.99 * (len(gaps) - 1))] if gaps else None, "tbt_max": gaps[-1] if gaps else None,
               "prompt_tokens": usage.prompt_tokens if usage else len(ctx),
               "completion_tokens": usage.completion_tokens if usage else None, "error": err}
        async with lock:
            out.write(json.dumps(rec) + "\n")
            out.flush()
        ctx = ctx + ids(rng, r["output"])  # the response comes back as input next round


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--max-ctx", type=int, default=36864)
    ap.add_argument("--compact-to", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = json.load(open(a.workload))
    client = AsyncOpenAI(base_url=a.url, api_key="x", timeout=7200, max_retries=0)
    sys_block = ids(random.Random(7), SYS_LEN)
    lock, t0 = asyncio.Lock(), time.time() + 1
    with open(a.out, "w") as out:
        await asyncio.gather(*(session(client, a, s, t0, out, lock, sys_block) for s in w["sessions"]))
    print(f"REPLAY done: {len(w['sessions'])} sessions in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
