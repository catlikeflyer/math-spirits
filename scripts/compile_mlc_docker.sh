#!/usr/bin/env bash
# scripts/compile_mlc_docker.sh
# Runs the MLC weight compilation pipeline inside a clean Linux container.
#
# Usage:
#   bash scripts/compile_mlc_docker.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Math Spirits: MLC Compilation via Docker ==="
echo "Repo root: $REPO_ROOT"
echo ""

docker run --platform linux/amd64 --rm \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  python:3.11-slim bash -c "
    echo 'Installing MLC tools...' && \
    pip install --default-timeout=1000 --retries 10 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch psutil && \
    pip install --default-timeout=1000 --retries 10 --no-cache-dir \
      https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_ai_nightly_cpu-0.26.dev246-py3-none-manylinux_2_28_x86_64.whl \
      https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_llm_nightly_cpu-0.26.dev6-py3-none-manylinux_2_28_x86_64.whl && \
    echo 'Patching dtype conversion compatibility...' && \
    sed -i 's/x.astype(dtype)/x.astype(str(dtype))/g' /usr/local/lib/python3.11/site-packages/mlc_llm/loader/standard_loader.py && \
    echo '' && \
    echo 'Running weight conversion...' && \
    bash scripts/compile_mlc.sh
"
