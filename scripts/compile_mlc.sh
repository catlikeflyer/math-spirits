#!/usr/bin/env bash
# scripts/compile_mlc.sh
# Converts Math Spirits fine-tuned weights to MLC format for WebGPU deployment.
#
# Pipeline per model:
#   1. mlc_llm convert_weight  → quantized MLC weight shards
#   2. mlc_llm gen_config      → mlc-chat-config.json
#   (wasm reused from prebuilt mlc-ai CDN — no compile step needed)
#
# Prerequisites:
#   conda activate mlc-env   (python 3.11 + mlc-llm installed)
#
# Output:
#   dist/mlc/math-ghost-1-q4f16_1-MLC/
#   dist/mlc/math-spectre-1-q4f16_1-MLC/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_ROOT="$REPO_ROOT/dist/mlc"
QUANTIZATION="q4f16_1"

echo "=== Math Spirits: MLC Weight Conversion Pipeline ==="
echo "Output dir: $DIST_ROOT"
echo "Quantization: $QUANTIZATION"
echo ""

# ─── Verify mlc_llm is available ─────────────────────────────────────────────
if ! python3 -c "import mlc_llm" &>/dev/null; then
  echo "[ERROR] mlc_llm not found. Install with:"
  echo "  conda activate mlc-env && pip install mlc-llm"
  exit 1
fi
echo "[OK] mlc_llm found: $(python3 -c "import mlc_llm; print(getattr(mlc_llm, '__version__', 'ready'))" 2>/dev/null || echo 'ready')"
echo ""

# ─── Helper function ─────────────────────────────────────────────────────────
compile_model() {
  local MODEL_NAME="$1"
  local FUSED_PATH="$2"
  local CONV_TEMPLATE="$3"
  local OUT_DIR="$DIST_ROOT/${MODEL_NAME}-${QUANTIZATION}-MLC"

  echo "──────────────────────────────────────────────────"
  echo "[MODEL] $MODEL_NAME"
  echo "[INPUT] $FUSED_PATH"
  echo "[OUTPUT] $OUT_DIR"
  echo ""

  if [ ! -d "$FUSED_PATH" ]; then
    echo "[ERROR] Fused model directory not found: $FUSED_PATH"
    echo "        Run scripts/fuse_spectre.sh first for spectre-1."
    return 1
  fi

  mkdir -p "$OUT_DIR"

  echo "[STEP 1/2] Converting weights (quantization: $QUANTIZATION)..."
  python3 -m mlc_llm convert_weight \
    "$FUSED_PATH" \
    --quantization "$QUANTIZATION" \
    --model-type qwen2 \
    -o "$OUT_DIR"

  echo ""
  echo "[STEP 2/2] Generating MLC chat config..."
  python3 -m mlc_llm gen_config \
    "$FUSED_PATH" \
    --quantization "$QUANTIZATION" \
    --conv-template chatml \
    --model-type qwen2 \
    -o "$OUT_DIR"

  echo ""
  echo "[OK] $MODEL_NAME conversion complete."
  echo "Artifacts:"
  ls -lh "$OUT_DIR"
  echo ""
}

# ─── math-ghost-1 (fused weights already exist) ──────────────────────────────
compile_model \
  "math-ghost-1" \
  "$REPO_ROOT/models/math-ghost-1/fused" \
  "chatml"

# ─── math-spectre-1 (run fuse_spectre.sh first) ──────────────────────────────
compile_model \
  "math-spectre-1" \
  "$REPO_ROOT/models/math-spectre-1/fused" \
  "chatml"

echo "══════════════════════════════════════════════════"
echo "[DONE] MLC conversion complete for both models."
echo ""
echo "Next step: run scripts/upload_hf.sh to push to HuggingFace."
