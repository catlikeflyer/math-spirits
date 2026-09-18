#!/usr/bin/env bash
# Package Fused math-ghost-1 to MLC WebGPU Weights
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUSED_DIR="${ROOT_DIR}/models/math-ghost-1/fused"
OUTPUT_DIR="${ROOT_DIR}/web/public/models/math-ghost-1-q4f16_1-MLC"

if [ ! -d "${FUSED_DIR}" ]; then
  echo "Error: Fused model directory does not exist at ${FUSED_DIR}."
  echo "Run mlx_lm.fuse first."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "==> Converting fused model to MLC WebGPU format (q4f16_1)..."
echo "Source: ${FUSED_DIR}"
echo "Destination: ${OUTPUT_DIR}"

# Check if mlc_llm is available locally
if command -v mlc_llm &> /dev/null; then
  mlc_llm convert_weight "${FUSED_DIR}" --quantization q4f16_1 -o "${OUTPUT_DIR}"
  mlc_llm gen_config "${FUSED_DIR}" --quantization q4f16_1 --conv-template qwen2 -o "${OUTPUT_DIR}"
elif command -v docker &> /dev/null; then
  echo "==> mlc_llm not found locally, executing via Docker (ghcr.io/mlc-ai/mlc-llm)..."
  docker run --rm \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    ghcr.io/mlc-ai/mlc-llm:latest \
    bash -c "mlc_llm convert_weight /workspace/models/math-ghost-1/fused --quantization q4f16_1 -o /workspace/web/public/models/math-ghost-1-q4f16_1-MLC && mlc_llm gen_config /workspace/models/math-ghost-1/fused --quantization q4f16_1 --conv-template qwen2 -o /workspace/web/public/models/math-ghost-1-q4f16_1-MLC"
else
  echo "Error: Neither mlc_llm nor Docker was found."
  echo "You can run this on Google Colab or Linux with:"
  echo "  pip install mlc-llm"
  echo "  mlc_llm convert_weight ./models/math-ghost-1/fused --quantization q4f16_1 -o ./web/public/models/math-ghost-1-q4f16_1-MLC"
  echo "  mlc_llm gen_config ./models/math-ghost-1/fused --quantization q4f16_1 --conv-template qwen2 -o ./web/public/models/math-ghost-1-q4f16_1-MLC"
  exit 1
fi

echo "==> Conversion complete! Weights ready for browser WebGPU at ${OUTPUT_DIR}"
