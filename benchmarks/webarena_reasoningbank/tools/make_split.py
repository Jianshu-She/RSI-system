"""Fixed train/test split of the 187 WebArena-Shopping tasks BY TEMPLATE (intent_template_id), so no template is in both.
Writes splits/{all,train,test}_ids.txt and splits/split.json.  Usage: python tools/make_split.py [--seed 0] [--test_frac 0.5]"""
import argparse, glob, json, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--test_frac", type=float, default=0.5)
a = ap.parse_args()
tasks = {}
for f in glob.glob(os.path.join(os.environ.get("WA_CODE", f"{ROOT}/work/reasoning-bank/WebArena"), "config_files", "*.json")):
    if not os.path.basename(f).split(".")[0].isdigit():
        continue
    c = json.load(open(f))
    if c["sites"] == ["shopping"]:
        tasks.setdefault(c["intent_template_id"], []).append(c["task_id"])
templates = sorted(tasks); random.Random(a.seed).shuffle(templates)
n_test = round(len(templates) * a.test_frac)
split = {"seed": a.seed, "test_frac": a.test_frac, "test_templates": sorted(templates[:n_test]), "train_templates": sorted(templates[n_test:])}
ids = {k: sorted(t for tpl in split[f"{k}_templates"] for t in tasks[tpl]) for k in ("train", "test")}
ids["all"] = sorted(ids["train"] + ids["test"])
os.makedirs(f"{ROOT}/splits", exist_ok=True)
for k, v in ids.items():
    open(f"{ROOT}/splits/{k}_ids.txt", "w").write("\n".join(map(str, v)) + "\n")
split.update({f"n_{k}": len(v) for k, v in ids.items()}); json.dump(split, open(f"{ROOT}/splits/split.json", "w"), indent=1)
print({k: len(v) for k, v in ids.items()}, "templates train/test:", len(split["train_templates"]), len(split["test_templates"]))
