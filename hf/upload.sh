#!/usr/bin/env bash
# Upload ARC-CT checkpoints to the Hugging Face Hub.
# Requires: pip install huggingface_hub && huggingface-cli login
set -euo pipefail
REPO=${REPO:-arc-ct/arc-ct}
SRC=${SRC:?set SRC to the directory holding the .pt checkpoints}

huggingface-cli repo create "$REPO" --type model -y || true
huggingface-cli upload "$REPO" hf/README.md README.md
for f in stage1_backbone.pt arcct_seed0.pt arcct_seed1.pt arcct_seed2.pt; do
  [ -f "$SRC/$f" ] && huggingface-cli upload "$REPO" "$SRC/$f" "$f"
done
echo "done: https://huggingface.co/$REPO"
