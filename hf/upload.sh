#!/usr/bin/env bash
# Upload ARC-CT checkpoints to the Hugging Face Hub.
#
#   pip install huggingface_hub
#   hf auth login
#   SRC=/path/to/checkpoints ./hf/upload.sh
#
# SRC must contain: stage1_backbone.pt arcct_seed{0,1,2}.pt
set -euo pipefail
REPO=${REPO:-arc-ct/arc-ct}
SRC=${SRC:?set SRC to the directory holding the .pt checkpoints}

hf repos create "$REPO" --type model --public || true
hf upload "$REPO" hf/README.md README.md --type model

for f in stage1_backbone.pt arcct_seed0.pt arcct_seed1.pt arcct_seed2.pt; do
  if [ -f "$SRC/$f" ]; then
    echo "uploading $f"
    hf upload "$REPO" "$SRC/$f" "$f" --type model
  else
    echo "missing (skipped): $SRC/$f"
  fi
done
echo "done: https://huggingface.co/$REPO"
