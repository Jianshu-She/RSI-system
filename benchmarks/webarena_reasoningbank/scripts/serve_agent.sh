#!/bin/bash
# Serve the agent model with vLLM.
# Usage: scripts/serve_agent.sh <hf_model_or_path> <served_name> <gpus> <port> [lora_adapter_path]
#   base / merged model : scripts/serve_agent.sh Qwen/Qwen3-32B qwen3-32b 0,1 8000
#   LoRA adapter        : scripts/serve_agent.sh Qwen/Qwen3-32B qwen3-32b-lora 0,1 8003 /path/to/adapter
#     (the adapter is exposed under <served_name>; the base under <served_name>-base)
# Qwen3-32B is served with YaRN x4 (131072 ctx) because agent prompts reach ~50k tokens; set EXTRA_VLLM_ARGS to change.
set -e
MODEL=$1; NAME=$2; GPUS=$3; PORT=$4; LORA=$5
[ -z "$PORT" ] && { sed -n 2,8p "$0"; exit 1; }
source "$(dirname "$0")/../env.sh" > /dev/null
source $WA_VLLM_VENV/bin/activate
TP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DEFAULT_ARGS='--max-model-len 131072 --hf-overrides {"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},"max_position_embeddings":131072}'
ARGS=(--tensor-parallel-size $TP --reasoning-parser qwen3 --enable-prefix-caching --host 127.0.0.1 --port $PORT)
if [ -n "$LORA" ]; then
  ARGS+=(--served-model-name "$NAME-base" --enable-lora --lora-modules "$NAME=$LORA" --max-lora-rank ${MAX_LORA_RANK:-64})
else
  ARGS+=(--served-model-name "$NAME")
fi
read -r -a EXTRA <<< "${EXTRA_VLLM_ARGS:-$DEFAULT_ARGS}"
CUDA_VISIBLE_DEVICES=$GPUS setsid nohup vllm serve "$MODEL" "${ARGS[@]}" "${EXTRA[@]}" > "$WA_LOGS/vllm_$NAME.log" 2>&1 < /dev/null &
echo "starting $NAME on GPUs $GPUS port $PORT (log: $WA_LOGS/vllm_$NAME.log)"
until curl -s -o /dev/null --max-time 5 http://127.0.0.1:$PORT/v1/models; do sleep 15; kill -0 $! 2>/dev/null || { echo "vLLM exited, see $WA_LOGS/vllm_$NAME.log"; exit 1; }; done
curl -s http://127.0.0.1:$PORT/v1/models | python3 -c "import json,sys; print('serving:', [m['id'] for m in json.load(sys.stdin)['data']])"
