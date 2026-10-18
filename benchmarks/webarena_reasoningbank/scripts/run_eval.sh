#!/bin/bash
# Launch one evaluation run (background, survives logout). Results -> runs/<name>/
#
# Usage: scripts/run_eval.sh --name NAME --method METHOD --model SERVED_NAME --tokenizer HF_ID --site PORT [options]
#   --method      no_memory | reasoningbank | reasoningbank_cross
#                   no_memory            plain agent (use this for base / LoRA models)
#                   reasoningbank        paper method: agent judges its own trajectory, agent writes memory
#                   reasoningbank_cross  judge = agreement with an independent 397B run (--cross_ref), 397B writes memory
#   --model       served model name at --agent_url (a LoRA adapter name works too)
#   --tokenizer   HF tokenizer id of the agent (the base model for LoRA)
#   --site        port of a FRESH shopping site (scripts/start_site.sh)
#   --split       test (default) | train | all          task list from splits/<split>_ids.txt
#   --agent_url   default http://127.0.0.1:8000/v1
#   --cross_ref   reference run dir for reasoningbank_cross (required for that method): an independent stronger-agent
#                 no_memory run over the same tasks, e.g. $WA_RUNS/ref_397b_all
#   --memory_dir  start ReasoningBank from an existing bank (default: empty bank in runs/NAME/memory)
#   --resume      continue a stopped run (skips tasks that already have results)
#   --max_new_tokens  per-step generation cap (default 65536 = paper code). Base Qwen3-32B never exceeded 5.7k tokens/step;
#                     use e.g. 8192 for models that can fall into repetition loops.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SPLIT=test; AGENT_URL=http://127.0.0.1:8000/v1; RESUME=""; MEMDIR=""; MAXTOK=65536
CROSS_REF=""
while [ $# -gt 0 ]; do case $1 in
  --name) NAME=$2; shift 2;; --method) METHOD=$2; shift 2;; --model) MODEL=$2; shift 2;; --tokenizer) TOK=$2; shift 2;;
  --site) SITE=$2; shift 2;; --split) SPLIT=$2; shift 2;; --agent_url) AGENT_URL=$2; shift 2;;
  --cross_ref) CROSS_REF=$2; shift 2;; --max_new_tokens) MAXTOK=$2; shift 2;; --memory_dir) MEMDIR=$2; shift 2;; --resume) RESUME=1; shift;;
  *) echo "unknown arg $1"; sed -n 2,20p "$0"; exit 1;; esac; done
for v in NAME METHOD MODEL TOK SITE; do [ -z "${!v}" ] && { echo "missing --${v,,}"; sed -n 2,20p "$0"; exit 1; }; done
source "$ROOT/env.sh" > /dev/null
OUT=$WA_RUNS/$NAME; IDS=$WA_HOME/splits/${SPLIT}_ids.txt
[ -f "$IDS" ] || { echo "no split file $IDS"; exit 1; }
if [ -d "$OUT" ] && [ -z "$RESUME" ]; then echo "$OUT exists; use --resume or a new --name"; exit 1; fi

export AGENT_MAX_NEW_TOKENS=$MAXTOK
export AGENT_LLM_URL=$AGENT_URL AGENT_LLM_MODEL=$MODEL AGENT_TOKENIZER=$TOK
export WA_SHOPPING=http://localhost:$SITE
source "$ROOT/env.sh" > /dev/null
# preflight: agent server serves MODEL, site answers, grader up
curl -s --max-time 10 "$AGENT_URL/models" | grep -q "\"$MODEL\"" || { echo "agent server at $AGENT_URL does not serve '$MODEL'"; exit 1; }
curl -s -o /dev/null --max-time 30 "http://localhost:$SITE/" || { echo "site on port $SITE not reachable"; exit 1; }
HITS=$(curl -s --max-time 120 "http://localhost:$SITE/catalogsearch/result/?q=shoe+storage" | grep -o 'toolbar-number">[0-9,]*' | tail -1 | grep -o '[0-9,]*$' || true)
[ "$HITS" = "19120" ] || { echo "site on port $SITE: product search broken ('$HITS' hits, expected 19120); refusing to evaluate"; exit 1; }
curl -s -o /dev/null --max-time 10 "$GT_JUDGE_URL/models" || { echo "grader at $GT_JUDGE_URL not up (scripts/serve_aux.sh)"; exit 1; }

mkdir -p "$OUT"      # only after the preflight checks passed
ARGS=(--website shopping --model "$MODEL" --task_ids_file "$IDS" --output_dir "$OUT")
case $METHOD in
  no_memory)           ARGS+=(--memory_mode no_memory);;
  reasoningbank)       ARGS+=(--memory_mode reasoningbank --judge autoeval);;
  reasoningbank_cross) [ -n "$CROSS_REF" ] || { echo "--cross_ref is required for reasoningbank_cross"; exit 1; }
                       ARGS+=(--memory_mode reasoningbank --judge autoeval --judge_model "$GT_JUDGE_MODEL" --cross_ref "$CROSS_REF");;
  *) echo "bad --method $METHOD"; exit 1;;
esac
[ -n "$MEMDIR" ] && ARGS+=(--memory_dir "$MEMDIR")
[ -n "$RESUME" ] && ARGS+=(--skip_done)

python3 - "$OUT/meta.json" "${ARGS[@]}" <<PY
import json, os, socket, sys, time
m = {"name": "$NAME", "method": "$METHOD", "model": "$MODEL", "tokenizer": "$TOK", "agent_url": "$AGENT_URL",
     "site": "http://localhost:$SITE", "split": "$SPLIT", "n_tasks": len(open("$IDS").read().split()),
     "cross_ref": "$CROSS_REF" if "$METHOD" == "reasoningbank_cross" else None, "memory_dir": "$MEMDIR" or None,
     "grader": os.environ["GT_JUDGE_MODEL"], "embedder": os.environ["EMBED_MODEL"], "host": socket.gethostname(),
     "agent_params": "temperature 0.7, max 30 steps, AX-tree obs, thinking on (code/run.py defaults)", "max_new_tokens": $MAXTOK,
     "command": "python pipeline_memory.py " + " ".join(sys.argv[2:]), "resumed": bool("$RESUME")}
key = "launches"; old = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
m[key] = old.get(key, []) + [time.strftime("%Y-%m-%d %H:%M:%S %Z")]
json.dump(m, open(sys.argv[1], "w"), indent=1)
PY
cd "$WA_CODE"
setsid nohup python pipeline_memory.py "${ARGS[@]}" >> "$OUT/pipeline.log" 2>&1 < /dev/null &
echo "launched $NAME (pid $!): $METHOD, model $MODEL, split $SPLIT ($(wc -w < $IDS) tasks), site :$SITE"
echo "progress: python $WA_HOME/tools/report.py $OUT     log: $OUT/pipeline.log"
