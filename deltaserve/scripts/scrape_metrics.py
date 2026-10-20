"""Poll a vLLM server's Prometheus /metrics every --interval s until --stop-file exists (or killed).

One JSONL line per poll: wall time + each metric summed over its label sets. KV-cache occupancy
(vllm:kv_cache_usage_perc), queue depth, prefix-cache counters, token counters.

    python3 scrape_metrics.py http://127.0.0.1:9300 out.jsonl [--interval 1] [--stop-file F]
"""
import argparse
import json
import os
import re
import time
import urllib.request

KEEP = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
        "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total", "vllm:num_preemptions_total", "vllm:prompt_tokens_cached_total")
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--stop-file", default="")
    a = ap.parse_args()
    with open(a.out, "a") as f:
        while not (a.stop_file and os.path.exists(a.stop_file)):
            t = time.time()
            try:
                txt = urllib.request.urlopen(a.url.rstrip("/") + "/metrics", timeout=5).read().decode()
                rec = {"t": t}
                for ln in txt.splitlines():
                    m = LINE.match(ln)
                    if m and m.group(1) in KEEP:
                        rec[m.group(1)] = rec.get(m.group(1), 0.0) + float(m.group(3))
                f.write(json.dumps(rec) + "\n")
                f.flush()
            except Exception:  # server not up yet / restarting
                pass
            time.sleep(max(0.0, a.interval - (time.time() - t)))


if __name__ == "__main__":
    main()
