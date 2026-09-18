#!/usr/bin/env bash
# scripts/fuse_spectre.sh
# Fuses math-spectre-1 LoRA adapters into the base Qwen2.5-Math-1.5B model.
# Output: models/math-spectre-1/fused/  (HF-compatible safetensors directory)
#
# Prerequisites:
#   pip install mlx-lm

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$REPO_ROOT/models/math-spectre-1"
ADAPTER_PATH="$MODEL_DIR/adapters"
OUTPUT_PATH="$MODEL_DIR/fused"

echo "=== Math Spirits: Fusing Spectre-1 LoRA Adapters ==="
echo "Base model : Qwen/Qwen2.5-Math-1.5B"
echo "Adapters   : $ADAPTER_PATH"
echo "Output     : $OUTPUT_PATH"
echo ""

if [ -d "$OUTPUT_PATH" ] && [ "$(ls -A "$OUTPUT_PATH")" ]; then
  echo "[INFO] Fused directory already exists and is non-empty."
  read -r -p "Re-fuse and overwrite? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Skipping fuse step."; exit 0; }
fi

mkdir -p "$OUTPUT_PATH"

echo "[STEP] Running mlx_lm.fuse..."
python3 -m mlx_lm.fuse \
  --model "Qwen/Qwen2.5-Math-1.5B" \
  --adapter-path "$ADAPTER_PATH" \
  --save-path "$OUTPUT_PATH" \
  --dequantize

echo ""
echo "[OK] Fusion complete. Output at: $OUTPUT_PATH"
ls -lh "$OUTPUT_PATH"
