"""ACE Finance "user traffic" against an OpenAI-compatible server, with traces.

Builds each request exactly like ACE's Generator (GENERATOR_PROMPT, temperature 0, thinking off),
streams it to measure TTFT / TBT (gaps between streamed chunks) / end-to-end latency and the server-side token
counts (prompt / completion), and scores it with ACE's own extract_answer +
DataProcessor.answer_is_correct. One JSONL record per request: the full prompt + response (the
trace), correctness, and timing. Run with the ACE venv python from the ACE upstream dir:

  cd $ACE_UPSTREAM_DIR && $ACE_PY $DS_HOME/scripts/rsi_traffic.py --url http://127.0.0.1:8000/v1 \
      --model llama3 --task formula --split train --context $PLAYBOOK \
      --concurrency 16 --out traces.jsonl
  # open-loop Poisson arrivals instead of a closed loop:  --rps 0.5 [--duration 600] [--loops 3]
  # + an FT job on the same server at t = --train-at s (open loop only; FT must be configured with a corpus):
  #   --train-mode co   POST /start_finetuning and keep serving (co-serving)
  #   --train-mode seq  stop serving: hold new arrivals, drain in-flight requests, train alone until --ft-final
  #                     exists, then release the held requests (time-disaggregated, one set of GPUs)
  #   Each record then has t_arrive and latency = done - arrival (incl. the hold); <out>.ft.json the FT times.
  # TraceLab-shaped arrivals instead of per-request Poisson:  --workload agent_workload.json (agent_trace.py)
  #   sessions arrive at their (Poisson) arrival times; each round waits its real TraceLab gap (tool latency / human
  #   think time), then sends the next ACE request and waits for its answer (closed loop within a session).
  #   Records then also carry session / round / task / kind.
  # Mixed traffic: --task formula,finer [--mix 1,2] interleaves the splits in that ratio (each request is scored
  #   with its own task's processor; records carry "task"). --duration also bounds --workload runs.
  # Several FT rounds instead of one job (rsi_mix.sh): --rounds-at 120,270,... --rounds-dir <rsi_mix_prep.py dir>
  #   round k starts at its time: DeltaServe (--ft-backend ds) round 0 = POST /start_finetuning (r0 is the launch
  #   corpus), later rounds POST /ft/ingest r<k>.jsonl; torchtune (--ft-backend tt) touch <rounds-dir>/go<k>.
  #   Round k is trained once the backend's cumulative input tokens reach rounds 0..k (meta.json): DeltaServe's
  #   --bwd-csv total_processed_tokens, torchtune's --tt-progress. seq holds traffic for each round.
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.getcwd())
from openai import AsyncOpenAI  # noqa: E402

from ace.prompts.generator import GENERATOR_PROMPT  # noqa: E402
from eval.finance.data_processor import DataProcessor  # noqa: E402
from utils import extract_answer  # noqa: E402


def load_samples(task, split):
    cfg = json.load(open("./eval/finance/data/sample_config.json"))[task]
    key = {"train": "train_data", "val": "val_data", "test": "test_data"}[split]
    proc = DataProcessor(task)
    return proc, proc.process_task_data([json.loads(l) for l in open(cfg[key])])


async def one(client, args, proc, playbook, i, s, out, lock, stats, tag=None):
    if isinstance(proc, tuple):  # mixed traffic: (task, processor)
        task_name, proc = proc
    else:
        task_name = args.task
    prompt = GENERATOR_PROMPT.format(playbook, "(empty)", s["question"], s["context"])
    t_arrive = time.time()
    await stats["gate"].wait()  # closed only while a sequential FT job holds the GPUs
    stats["inflight"] += 1
    t0 = time.time()
    ttft = None
    parts, t_chunks, usage = [], [], None
    try:
        stream = await client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=args.max_tokens, stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        async for ch in stream:
            if ch.usage:
                usage = ch.usage
            if ch.choices and ch.choices[0].delta.content:
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                t_chunks.append(now)
                parts.append(ch.choices[0].delta.content)
        err = None
    except Exception as e:  # keep the run alive; the record says what failed
        err = repr(e)
    e2e = time.time() - t0
    stats["inflight"] -= 1
    resp = "".join(parts)
    ans = extract_answer(resp) if resp else None
    correct = bool(resp) and proc.answer_is_correct(ans, s["target"])
    gaps = [b - a for a, b in zip(t_chunks, t_chunks[1:])]
    rec = {"index": i, "task": task_name, "split": args.split, "t_arrive": t_arrive, "t_send": t0,
           "latency": t0 + e2e - t_arrive, "ttft": ttft,
           "e2e": e2e, "tbt_mean": sum(gaps) / len(gaps) if gaps else None, "tbt_max": max(gaps) if gaps else None,
           "tbt_gaps": [round(g, 5) for g in gaps] if args.save_gaps else None,
           "prompt_tokens": usage.prompt_tokens if usage else None,
           "completion_tokens": usage.completion_tokens if usage else None, "correct": correct, "error": err, "prompt": prompt, "response": resp,
           "question": s["question"], "context": s["context"], "target": s["target"], **(tag or {})}
    async with lock:
        out.write(json.dumps(rec) + "\n")
        out.flush()
        stats["n"] += 1
        stats["ok"] += correct


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--context", default="", help="playbook file (empty = '(empty)')")
    ap.add_argument("--concurrency", type=int, default=0, help="closed loop with N workers")
    ap.add_argument("--rps", type=float, default=0.0, help="open-loop Poisson arrival rate")
    ap.add_argument("--loops", type=int, default=1, help="passes over the split (open loop)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop sending after N s (open loop)")
    ap.add_argument("--stop-file", default="", help="stop sending once this file exists (open loop)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-gaps", type=int, default=1, help="store every inter-chunk gap (for TBT percentiles)")
    ap.add_argument("--train-at", type=float, default=-1, help="start the FT job at this many s (open loop)")
    ap.add_argument("--train-mode", choices=["co", "seq"], default="co")
    ap.add_argument("--ft-final", default="", help="file whose existence means the FT job is done")
    ap.add_argument("--workload", default="", help="agent_trace.py JSON: TraceLab session timing (open loop only)")
    ap.add_argument("--mix", default="", help="per-task interleave ratio for --task a,b (default 1 each)")
    ap.add_argument("--rounds-at", default="", help="comma-separated start times (s) of the FT rounds")
    ap.add_argument("--rounds-dir", default="")
    ap.add_argument("--ft-backend", choices=["ds", "tt"], default="ds")
    ap.add_argument("--bwd-csv", default="")
    ap.add_argument("--tt-progress", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks_ = args.task.split(",")
    if len(tasks_) == 1:
        proc, samples = load_samples(args.task, args.split)
        if args.limit:
            samples = samples[: args.limit]
        procs = [proc] * len(samples)
    else:  # interleave the splits in the --mix ratio until each is used up
        loaded = [load_samples(t, args.split) for t in tasks_]
        ratio = [int(x) for x in args.mix.split(",")] if args.mix else [1] * len(tasks_)
        pos, samples, procs = [0] * len(tasks_), [], []
        while any(pos[j] < len(loaded[j][1]) for j in range(len(tasks_))):
            for j, t in enumerate(tasks_):
                for _ in range(ratio[j]):
                    if pos[j] < len(loaded[j][1]):
                        samples.append(loaded[j][1][pos[j]])
                        procs.append((t, loaded[j][0]))
                        pos[j] += 1
        if args.limit:
            samples, procs = samples[: args.limit], procs[: args.limit]
    playbook = open(args.context).read() if args.context else "(empty)"
    client = AsyncOpenAI(base_url=args.url, api_key="x", timeout=3600)
    lock, stats = asyncio.Lock(), {"n": 0, "ok": 0, "gate": asyncio.Event(), "inflight": 0}
    stats["gate"].set()
    t_start = time.time()
    ft = {"mode": args.train_mode, "t_start": t_start}

    async def post(path, body=None):
        base = args.url.rsplit("/v1", 1)[0]
        req = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return await asyncio.to_thread(lambda: urllib.request.urlopen(req, timeout=600).read())

    async def train_job():
        await asyncio.sleep(args.train_at)
        ft["t_trigger"] = time.time()
        if args.train_mode == "seq":
            stats["gate"].clear()
            while stats["inflight"]:
                await asyncio.sleep(0.05)
            ft["t_drained"] = time.time()
        await post("/start_finetuning")
        ft["t_ft_start"] = time.time()
        if args.train_mode == "seq":  # no traffic: wake the idle engine loop so FT admission runs
            await post("/v1/completions", {"model": args.model, "prompt": "hi", "max_tokens": 2})
        while not os.path.exists(args.ft_final):
            await asyncio.sleep(0.5)
        ft["t_ft_end"] = time.time()
        stats["gate"].set()
    def trained_tokens():
        try:
            if args.ft_backend == "ds":
                last = open(args.bwd_csv).read().strip().splitlines()[-1].split(",")
                hdr = open(args.bwd_csv).readline().strip().split(",")
                return int(last[hdr.index("total_processed_tokens")])
            return int(open(args.tt_progress).read().strip().splitlines()[-1].split()[1])
        except (OSError, IndexError, ValueError):
            return 0

    async def rounds_job():
        meta = json.load(open(f"{args.rounds_dir}/meta.json"))["rounds"]
        at = [float(x) for x in args.rounds_at.split(",")]
        ft["rounds"] = []
        cum = 0
        for k, t_k in enumerate(at):
            cum += meta[k]["tokens"]
            rd = {"k": k, "tokens": meta[k]["tokens"], "target_cum": cum}
            ft["rounds"].append(rd)
            await asyncio.sleep(max(0.0, t_start + t_k - time.time()))
            rd["t_trigger"] = time.time()
            if args.train_mode == "seq":
                stats["gate"].clear()
                while stats["inflight"]:
                    await asyncio.sleep(0.05)
                rd["t_drained"] = time.time()
            if args.ft_backend == "tt":
                open(f"{args.rounds_dir}/go{k}", "w").close()
            elif k == 0:
                await post("/start_finetuning")
            else:
                rows = [json.loads(l) for l in open(f"{args.rounds_dir}/r{k}.jsonl")]
                rd["ingest"] = json.loads(await post("/ft/ingest", {"rows": rows}))
            rd["t_ft_start"] = time.time()
            if args.ft_backend == "ds":  # wake an idle engine loop so FT admission runs
                await post("/v1/completions", {"model": args.model, "prompt": "hi", "max_tokens": 2})
            if args.train_mode == "seq":
                while trained_tokens() < cum:
                    await asyncio.sleep(0.2)
                rd["t_done"] = time.time()
                stats["gate"].set()
        while True:  # co: record when each round's tokens are trained
            n = trained_tokens()
            for rd in ft["rounds"]:
                if "t_done" not in rd and n >= rd["target_cum"]:
                    rd["t_done"] = time.time()
            if all("t_done" in rd for rd in ft["rounds"]):
                break
            await asyncio.sleep(0.2)
        ft["t_ft_end"] = ft["rounds"][-1]["t_done"]

    async def co_round_monitor():  # in co the rounds overlap: mark done times while later rounds are pending
        while "rounds" not in ft or not all("t_done" in rd for rd in ft["rounds"]):
            n = trained_tokens()
            for rd in ft.get("rounds", []):
                if "t_done" not in rd and "t_trigger" in rd and n >= rd["target_cum"]:
                    rd["t_done"] = time.time()
            await asyncio.sleep(0.2)
    if args.rounds_at:
        trainer = asyncio.create_task(rounds_job())
        if args.train_mode == "co":
            asyncio.create_task(co_round_monitor())
    else:
        trainer = asyncio.create_task(train_job()) if args.train_at >= 0 else None
    with open(args.out, "a") as out:
        if args.concurrency:
            sem = asyncio.Semaphore(args.concurrency)

            async def bounded(i, s):
                async with sem:
                    await one(client, args, procs[i], playbook, i, s, out, lock, stats)
            await asyncio.gather(*(bounded(i, s) for i, s in enumerate(samples)))
        elif args.workload:
            nxt = iter(range(len(samples)))  # every ACE request is sent once, in order, across the sessions
            used_up = asyncio.Event()  # the last request is out: waiting sessions stop instead of sleeping on

            async def nap(dt):
                if args.duration:  # never sleep past the end of sending
                    dt = min(dt, t_start + args.duration - time.time())
                try:
                    await asyncio.wait_for(used_up.wait(), max(0.0, dt))
                except asyncio.TimeoutError:
                    pass

            async def agent_session(w):
                if args.duration and w["arrival"] > args.duration:
                    return
                await nap(t_start + w["arrival"] - time.time())
                task = -1
                for k, r in enumerate(w["rounds"]):
                    await nap(r["wait"])
                    if args.duration and time.time() - t_start > args.duration:
                        return
                    i = None if used_up.is_set() else next(nxt, None)
                    if i is None:
                        return
                    if i == len(samples) - 1:
                        used_up.set()
                    task += r["kind"] == "user"
                    await one(client, args, procs[i], playbook, i, samples[i], out, lock, stats,
                              {"session": w["id"], "round": k, "agent_task": task, "kind": r["kind"]})
            await asyncio.gather(*(agent_session(w) for w in json.load(open(args.workload))["sessions"]))
        else:
            rng = random.Random(args.seed)
            tasks = []
            for _ in range(max(1, args.loops)):
                for i, s in enumerate(samples):
                    if args.duration and time.time() - t_start > args.duration:
                        break
                    if args.stop_file and os.path.exists(args.stop_file):
                        break
                    tasks.append(asyncio.create_task(one(client, args, procs[i], playbook, i, s, out, lock, stats)))
                    await asyncio.sleep(rng.expovariate(args.rps))
            await asyncio.gather(*tasks)
    if trainer:
        await trainer
        ft["t_traffic_end"] = max(json.loads(l)["t_arrive"] + json.loads(l)["latency"] for l in open(args.out))
        json.dump(ft, open(args.out + ".ft.json", "w"), indent=1)
    print(f"RESULT {args.task} {args.split}: {stats['ok']}/{stats['n']} correct "
          f"({100 * stats['ok'] / max(1, stats['n']):.1f}%), {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
