#!/usr/bin/env bash
# scripts/upload_hf.sh
# Uploads MLC-compiled model artifacts to HuggingFace Hub.
#
# Usage:
#   HF_USERNAME=your-username bash scripts/upload_hf.sh
#
# Prerequisites:
#   pip install huggingface_hub
#   hf auth login (or huggingface-cli login)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_ROOT="$REPO_ROOT/dist/mlc"

# ─── Require HF_USERNAME ─────────────────────────────────────────────────────
if [ -z "${HF_USERNAME:-}" ]; then
  echo "[ERROR] HF_USERNAME is not set."
  echo "Usage: HF_USERNAME=your-username bash scripts/upload_hf.sh"
  exit 1
fi

echo "=== Math Spirits: HuggingFace Upload ==="
echo "HF username : $HF_USERNAME"
echo "Source      : $DIST_ROOT"
echo ""

# ─── Verify login via python huggingface_hub ────────────────────────────────
if ! python3 -c "
import huggingface_hub
from huggingface_hub import HfApi
api = HfApi()
try:
    user = api.whoami()
    print(f'[OK] Authenticated as: {user.get(\"name\", user.get(\"username\", \"unknown\"))}')
except Exception as e:
    import sys
    print(f'[ERROR] Not logged in to Hugging Face: {e}')
    print('Please run: hf auth login')
    sys.exit(1)
" 2>&1; then
  exit 1
fi
echo ""

# ─── Helper function ─────────────────────────────────────────────────────────
upload_model() {
  local MODEL_DIR="$1"
  local REPO_NAME="$2"
  local FULL_REPO="$HF_USERNAME/$REPO_NAME"

  if [ ! -d "$MODEL_DIR" ]; then
    echo "[SKIP] $MODEL_DIR not found. Run compile_mlc_docker.sh first."
    return 0
  fi

  echo "──────────────────────────────────────────────────"
  echo "[UPLOAD] $MODEL_DIR → hf.co/$FULL_REPO"

  # Create repo if it doesn't exist
  python3 -c "
from huggingface_hub import HfApi
import sys
api = HfApi()
try:
    api.create_repo('$FULL_REPO', repo_type='model', exist_ok=True, private=False)
    print('  Repo ready: https://huggingface.co/$FULL_REPO')
except Exception as e:
    print(f'  [ERROR] Failed to create repo {e}')
    print('  Please ensure you are logged in with a WRITE token via: hf auth login --force')
    sys.exit(1)
"

  # Upload the full directory
  python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$MODEL_DIR',
    repo_id='$FULL_REPO',
    repo_type='model',
    commit_message='Add MLC q4f16_1 WebGPU weights via Math Spirits pipeline',
)
print('  Upload complete: https://huggingface.co/$FULL_REPO')
"
  echo ""
}

# ─── Upload both models ───────────────────────────────────────────────────────
upload_model \
  "$DIST_ROOT/math-ghost-1-q4f16_1-MLC" \
  "math-ghost-1-q4f16_1-MLC"

upload_model \
  "$DIST_ROOT/math-spectre-1-q4f16_1-MLC" \
  "math-spectre-1-q4f16_1-MLC"

echo "══════════════════════════════════════════════════"
echo "[DONE] Upload complete."
echo ""
echo "Ghost-1  : https://huggingface.co/$HF_USERNAME/math-ghost-1-q4f16_1-MLC"
echo "Spectre-1: https://huggingface.co/$HF_USERNAME/math-spectre-1-q4f16_1-MLC"
echo ""
