"""Dump, per task: eval types, reference answers, agent's final message, last action, GT reward, n_steps.
Usage: python audit_answers.py <run_dir> > out.jsonl"""
import gzip, json, os, pickle, re, sys
CFG = os.path.join(os.environ.get("WA_CODE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "reasoning-bank", "WebArena")), "config_files")
rd = sys.argv[1]
for d in sorted((x for x in os.listdir(rd) if x.startswith("webarena.")), key=lambda x: int(x.split(".")[1])):
    p = os.path.join(rd, d)
    if not os.path.exists(f"{p}/summary_info.json"):
        continue
    tid = int(d.split(".")[1])
    s = json.load(open(f"{p}/summary_info.json"))
    cfg = json.load(open(f"{CFG}/{tid}.json"))
    steps = sorted((f for f in os.listdir(p) if f.startswith("step_")), key=lambda f: int(re.findall(r"\d+", f)[0]))
    actions, last_url = [], None
    for f in steps:
        st = pickle.load(gzip.open(f"{p}/{f}"))
        if st.action:
            actions.append(st.action)
        if st.obs and "url" in st.obs:
            last_url = st.obs["url"]
    msg = [a for a in actions if "send_msg_to_user" in a]
    print(json.dumps({
        "tid": tid, "reward": s.get("cum_reward"), "n_steps": s.get("n_steps"),
        "intent": cfg["intent"], "eval_types": cfg["eval"]["eval_types"],
        "ref": cfg["eval"]["reference_answers"], "ref_url": cfg["eval"]["reference_url"],
        "program_html": cfg["eval"]["program_html"],
        "final_msg": msg[-1] if msg else None, "last_action": actions[-1] if actions else None,
        "last_url": last_url,
    }))
