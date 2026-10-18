#!/bin/bash
# Build $MODEL_DIR = Qwen3-32B with YaRN x4 (131,072-token context) WITHOUT copying weights:
#   every file of the HF snapshot is symlinked, except config.json, which is rewritten with
#     "max_position_embeddings": 131072,
#     "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
#   (Qwen3 model-card recipe; vLLM 0.31 did not apply YaRN via --hf-overrides, so a local dir is used.)
# Source snapshot: $QWEN3_32B_SRC if set, else downloaded with huggingface_hub (revision used in the paper reproduction).
# Usage: setup/make_yarn_model.sh            (needs a python with huggingface_hub, default $VLLM_PY)
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../env.sh"
REV=9216db5781bf21249d130ec9da846c4624c16137   # Qwen/Qwen3-32B snapshot used by the source runs
PY=${MODEL_PY:-$VLLM_PY}; command -v "$PY" >/dev/null 2>&1 || PY=python3

SRC=${QWEN3_32B_SRC:-}
if [ -z "$SRC" ]; then
  SRC=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-32B', revision='$REV'))")
fi
[ -f "$SRC/config.json" ] || { echo "no config.json in $SRC"; exit 1; }
echo "source snapshot: $SRC"
mkdir -p "$MODEL_DIR"
for f in "$SRC"/*; do
  b=$(basename "$f"); [ "$b" = config.json ] && continue
  ln -sfn "$(readlink -f "$f")" "$MODEL_DIR/$b"
done
python3 - "$SRC/config.json" "$MODEL_DIR/config.json" <<'EOF'
import json, sys
src, dst = sys.argv[1:3]
c = json.load(open(src))
c["max_position_embeddings"] = 131072
c["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
if dst and __import__("os").path.islink(dst):
    __import__("os").remove(dst)
json.dump(c, open(dst, "w"), indent=2)
EOF
echo "built $MODEL_DIR:"; ls -l "$MODEL_DIR" | head -5
