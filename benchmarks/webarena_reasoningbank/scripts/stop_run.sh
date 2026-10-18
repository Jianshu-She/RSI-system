#!/bin/bash
# Stop a running evaluation (pipeline + its current task).  Usage: scripts/stop_run.sh <name>
source "$(dirname "$0")/../env.sh" > /dev/null; OUT=$WA_RUNS/$1
[ -z "$1" ] && { echo "usage: $0 <name>"; exit 1; }
for p in $(pgrep -u $(id -u) -f "[p]ipeline_memory.py .*--output_dir $OUT( |$)"); do kill $p && echo "killed pipeline $p"; done
for p in $(pgrep -u $(id -u) -f "[r]un.py .*--results_path $OUT( |$)|[c]ross_check .*$OUT/|[e]valuate_trajectory .*$OUT/|[i]nduce_memory.py .*$OUT( |$)"); do kill $p && echo "killed $p"; done
echo "unfinished task dirs (delete or ignore; --resume reruns them):"; ls -d $OUT/20* 2>/dev/null
