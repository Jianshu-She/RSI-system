#!/bin/bash
#SBATCH --job-name=wa-finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --output=%x_%j.out
# Build the SFT set from saved runs (TRAIN split only) and train a LoRA adapter (data parallel over the allocated GPUs).
# Usage: TEACHER="$WA_RUNS/teacher397_train_s1 ..." SELF="$WA_RUNS/base_q32_all" OUT_NAME=lora_v3 \
#        sbatch --account=<acct> --qos=<qos> --partition=<part> jobs/finetune.sbatch
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"; source env.sh > /dev/null
OUT_NAME=${OUT_NAME:-lora}; TRAIN_VENV=${TRAIN_VENV:-$WA_WORK/envs/train}
DATA=$WA_WORK/sft/$OUT_NAME.jsonl; CKPT=$WA_WORK/ckpt/$OUT_NAME
python datagen/build_sft_data.py --out "$DATA" ${TEACHER:+--teacher $TEACHER} ${SELF:+--self $SELF}
NGPU=$(nvidia-smi -L | wc -l)
"$TRAIN_VENV/bin/torchrun" --nproc_per_node=$NGPU "$WA_HOME/../../rsi/finetune/train_lora.py" \
  --base Qwen/Qwen3-32B --data "$DATA" --out "$CKPT" --epochs ${EPOCHS:-1} --lr ${LR:-1e-4} --rank 16 --alpha 32 \
  --batch 16 --loss_on last --thinking on --max_len 24576
echo "adapter: $CKPT/final   (evaluate: ADAPTER=$CKPT/final sbatch jobs/eval.sbatch $OUT_NAME)"
