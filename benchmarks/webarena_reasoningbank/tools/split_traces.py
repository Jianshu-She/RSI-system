"""Split an exported trace JSONL into K files of ~equal size; a trajectory (run, task_id) never spans two files.
Usage: python split_traces.py <in.jsonl> <out_dir> [K=10]"""
import collections, json, os, sys
src, out, K = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10
groups = collections.OrderedDict()
for line in open(src):
    r = json.loads(line)
    groups.setdefault((r["run"], r["task_id"]), []).append(line)
sizes, parts = [0] * K, [[] for _ in range(K)]
for key, lines in sorted(groups.items(), key=lambda kv: -sum(map(len, kv[1]))):
    i = sizes.index(min(sizes)); parts[i].append((key, lines)); sizes[i] += sum(map(len, lines))
os.makedirs(out, exist_ok=True)
for i, p in enumerate(parts):
    with open(f"{out}/all_steps_part{i:02d}.jsonl", "w") as f:
        for _, lines in sorted(p):
            f.writelines(lines)
    print(f"part{i:02d}: {len(p)} trajectories, {sum(len(l) for _, l in p)} steps, {sizes[i]/1e6:.1f} MB")
