---
license: mit
tags:
  - medical-imaging
  - computed-tomography
  - vision-language
  - contrastive-learning
  - zero-shot-classification
datasets:
  - ibrahimhamamci/CT-RATE
library_name: pytorch
pipeline_tag: zero-shot-classification
---

# ARC-CT — Anatomy-Routed Contrastive Vision-Language Learning for 3D Chest CT

Checkpoints for [ARC-CT](https://github.com/arc-ct/arc-ct) (TIA workshop @ MICCAI 2026, oral).
ARC-CT learns from paired chest CT volumes and radiology reports — no manual
annotations, no bounding boxes — and classifies 18 abnormalities zero-shot by
comparing a volume against text prompts.

## Files

| file | what it is |
|---|---|
| `stage1_backbone.pt` | Stage-1 supervised 3D ResNet-18 (warm-start for Stage 2) |
| `arcct_seed0.pt` | full ARC-CT model, seed 0 |
| `arcct_seed1.pt` | full ARC-CT model, seed 1 |
| `arcct_seed2.pt` | full ARC-CT model, seed 2 |

Reported numbers are the three-seed mean; any single seed reproduces to within
the seed-to-seed s.d. of ~0.001 macro AUC.

## Usage

```bash
git clone https://github.com/arc-ct/arc-ct && cd arc-ct
pip install -r requirements.txt

python -c "from huggingface_hub import snapshot_download; \
           snapshot_download('arc-ct/arc-ct', local_dir='checkpoints')"

RAC_EVAL_CKPT=checkpoints/arcct_seed0.pt python tools/evaluate.py
```

## Model

- **Image encoder** 3D ResNet-18 (Kinetics-400 init), input `3 x 96 x 192 x 192`
  at 1.5 x 1.5 x 3.0 mm, three HU windows (lung / soft tissue / bone).
- **AnatomyQFormer** 30 queries — 10 anatomy (each masked to one organ group),
  18 pathology, 2 global.
- **Text encoder** CXR-BERT, frozen except LoRA (rank 8) on the top layers'
  query/value projections.
- **Objectives** label-Jaccard soft InfoNCE, per-organ alignment, per-token query
  supervision.

Inference needs no organ masks and no language-model calls.

## Intended use and limits

Research only. Not a medical device, not for clinical decision-making. Trained on
CT-RATE (non-contrast chest CT, single source); accuracy drops under scanner and
protocol shift, as reported for the external RAD-ChestCT evaluation in the paper.

## Citation

```bibtex
@inproceedings{isik2026arcct,
  title     = {{ARC}-{CT}: Anatomy-Routed Contrastive Vision-Language Learning for 3D Chest {CT}},
  author    = {Huseyin Umut Isik and Mehmet Alp Ozaydin and Sila Kurugol and Seyda Ertekin},
  booktitle = {The Third International Workshop on Thoracic Image Analysis at MICCAI 2026},
  year      = {2026},
  url       = {https://openreview.net/forum?id=t6mhjZ4R70}
}
```
