"""Export saved WebArena trajectories to JSONL (one record per agent step) for SFT / LoRA comparisons.
Each record has the exact chat the agent saw (system + user prompt incl. AX tree, history, retrieved memory) and the
model's full output (<think>...</think> + <action>), plus task-level labels (GT reward, self-judge) for filtering.
Usage: python export_traces.py --out traces/all_steps.jsonl results/shop_nomem_q32 results/shop_rb_q32 ...
"""
import argparse, gzip, json, os, pickle, re

CFG = os.path.join(os.environ.get("WA_CODE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "reasoning-bank", "WebArena")), "config_files")


def to_chat(cm):
    """agent_info["chat_messages"] is a list of strings in fixed order: [system, user, assistant] or, after parse
    retries, [system, user, assistant(bad), user(retry error), ..., assistant(accepted)] (agents/legacy/agent.py)."""
    return [{"role": "system" if i == 0 else ("user" if i % 2 == 1 else "assistant"), "content": c}
            for i, c in enumerate(cm)]


def export_run(rd, f):
    n = 0
    run = os.path.basename(rd.rstrip("/"))
    for d in sorted(x for x in os.listdir(rd) if x.startswith("webarena.")):
        p = os.path.join(rd, d)
        if not os.path.exists(f"{p}/summary_info.json"):
            continue
        tid = int(d.split(".")[1])
        s = json.load(open(f"{p}/summary_info.json"))
        cfg = json.load(open(f"{CFG}/{tid}.json"))
        judge = {}
        for j in (x for x in os.listdir(p) if x.endswith("_autoeval.json")):
            r = json.load(open(f"{p}/{j}"))[0]
            judge[j[:-len("_autoeval.json")]] = r.get("rm")
        steps = sorted((x for x in os.listdir(p) if re.match(r"step_\d+\.pkl\.gz", x)), key=lambda x: int(re.findall(r"\d+", x)[0]))
        for x in steps:
            st = pickle.load(gzip.open(f"{p}/{x}"))
            ai = st.agent_info or {}
            cm = ai.get("chat_messages") or []
            if not st.action or len(cm) < 3:
                continue
            chat = to_chat(cm)
            f.write(json.dumps({
                "run": run, "task_id": tid, "step": int(re.findall(r"\d+", x)[0]),
                "intent": cfg["intent"], "intent_template_id": cfg["intent_template_id"],
                "messages": chat[:-1],                          # model input (system, user[, retry turns])
                "response": chat[-1]["content"],                # accepted model output (<think> + <action>)
                "n_parse_retries": (len(cm) - 3) // 2,
                # well-formed target: <think>...</think> followed by a complete <action>...</action>
                "response_format_ok": "</think>" in chat[-1]["content"] and "<action>" in chat[-1]["content"].rsplit("</think>", 1)[-1]
                                      and "</action>" in chat[-1]["content"].rsplit("</think>", 1)[-1],
                "has_memory_in_prompt": "memory items that I accumulated" in cm[0],
                "action": st.action, "url": (st.obs or {}).get("url"),
                "task_gt_reward": s.get("cum_reward"), "task_n_steps": s.get("n_steps"),
                "task_err": s.get("err_msg"), "task_self_judge": judge,
            }) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for rd in a.run_dirs:
            print(rd, export_run(rd, f), "steps")
