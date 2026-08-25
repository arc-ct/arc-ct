# Reproducing the paper

Hardware used: one 80 GB A100. Stage 1 is ~8 h, Stage 2 ~30 h per seed.
All reported ARC-CT numbers are the mean over `RAC_SEED=0,1,2`.

## Table 1 / Table 2 — CT-RATE and RAD-ChestCT

Prerequisite: build the Qwen3-8B organ-sentence cache once. Stage 2's per-organ
alignment trains on it; without it the loader falls back to keyword extraction and
the numbers below do not reproduce.

```bash
RAC_REPORTS_CSV=DATA/ct_reports/train_reports.csv \
RAC_REGION_OUT=DATA/region_cache_qwen3.json \
python tools/extract_organ_sentences.py

set -a; source configs/stage1.env; set +a
RAC_DATA_ROOT=DATA RAC_STAGE1_RESULTS_DIR=runs/stage1 python tools/train_stage1.py

for SEED in 0 1 2; do
  set -a; source configs/stage2.env; set +a
  RAC_DATA_ROOT=DATA RAC_STAGE1_CKPT=runs/stage1/stage1_best.pt \
  RAC_REGION_CACHE=DATA/region_cache_qwen3.json \
  RAC_SEED=$SEED RAC_RESULTS_DIR=runs/arcct_seed$SEED python tools/train_stage2.py

  RAC_EVAL_CKPT=runs/arcct_seed$SEED/CTCLIP.best.pt python tools/evaluate.py     # in-domain
  RAC_EVAL_CKPT=runs/arcct_seed$SEED/CTCLIP.best.pt python tools/eval_radchest.py # external
done
```

`tools/evaluate.py` prints macro AUC, the 18 per-class AUCs (Table 2), and
support-weighted F1 / positive-class precision / accuracy following the CT-CLIP
protocol.

### Masks at inference

Stage 2 trains with TotalSegmentator masks, and the evaluation script can route
the Q-Former with masks or run mask-free. `RAC_MASK_VALID` controls this: point
it at a mask directory to route with masks, leave it empty for the mask-free
path that the deployed model uses. The two differ, and the difference is
concentrated in small/focal findings — report which one you are quoting.

## Table 3 — ablations

Single-flag overrides on top of `configs/stage2.env`; see `configs/ablations.md`.
Confidence intervals come from a paired patient-clustered bootstrap over the
validation set (resample patients, recompute macro AUC, 10,000 draws).

## Table 4 — retrieval

```bash
RAC_EVAL_CKPT=runs/arcct_seed0/CTCLIP.best.pt python tools/eval_retrieval.py
```

Retrieval reuses the classification embedding; there is no retrieval-specific
training or tuning.

## Figure 3 — Grad-CAM

```bash
python tools/gradcam.py --ckpt runs/arcct_seed0/CTCLIP.best.pt \
    --volume DATA/npz/valid/<volume>.npz --classes "Cardiomegaly,Arterial wall calcification,Pleural effusion,Consolidation"
```

## Sanity check (no data required)

```bash
python - <<'EOF'
import os, torch; os.environ["RAC_KINETICS_REQUIRED"] = "0"
from arcct.image_encoder import RACImageEncoder
from arcct.anatomy_qformer import AnatomyQFormer
from arcct.dataset import PATHOLOGIES, PATHOLOGY_FINE_ORGANS
enc = RACImageEncoder(pretrained_kinetics=False).eval()
feat = enc.forward_spatial(torch.randn(1, 3, 96, 192, 192))     # -> (1, 512, 12, 12, 12)
qf = AnatomyQFormer(anatomy_labels=list(range(1, 11)),
                    pathology_labels=[PATHOLOGY_FINE_ORGANS.get(p, []) for p in PATHOLOGIES],
                    num_global=2, dim=512, depth=4, num_heads=8, image_dim=512).eval()
print(feat.shape, qf.qformer.num_queries)                        # (1,512,12,12,12) 30
EOF
```
