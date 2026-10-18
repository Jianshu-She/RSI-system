#!/bin/bash
#SBATCH --job-name=wa-collect
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x_%j.out
# Data generation: run a teacher agent (default Qwen3.5-397B, which is also the grader) on the TRAIN split, N times in
# parallel on N fresh sites. Output: $WA_RUNS/<PREFIX>_s1..sN, each a resumable no_memory run.
# Usage:  sbatch --account=<acct> --qos=<qos> --partition=<part> jobs/collect_teacher.sbatch [N=3] [PREFIX=teacher397_train]
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source env.sh > /dev/null
N=${1:-3}; PREFIX=${2:-teacher397_train}
NO_EMBED=1 bash scripts/serve_aux.sh                        # 397B on GPU4-7 :8002 (agent + grader)
for k in $(seq 1 $N); do
  bash scripts/start_site.sh "site_$PREFIX$k" $k            # fails loudly if the site's search does not work
  R=""; [ -d "$WA_RUNS/${PREFIX}_s$k" ] && R=--resume
  bash scripts/run_eval.sh $R --name "${PREFIX}_s$k" --method no_memory --model "$GT_JUDGE_MODEL" \
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 --agent_url "$GT_JUDGE_URL" --site $((7770+k)) --split train
done
sleep 60; while pgrep -u $(id -u) -f "[p]ipeline_memory.py .*runs/${PREFIX}_s" > /dev/null; do sleep 120; done
python tools/report.py --split train $(ls -d $WA_RUNS/${PREFIX}_s*)
