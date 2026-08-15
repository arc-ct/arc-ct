# ARC-CT: Anatomy-Routed Contrastive Vision-Language Learning for 3D Chest CT

[![paper](https://img.shields.io/badge/paper-OpenReview-b31b1b)](https://openreview.net/pdf?id=t6mhjZ4R70)
[![weights](https://img.shields.io/badge/weights-HuggingFace-yellow)](https://huggingface.co/arc-ct/arc-ct)

Official implementation of *ARC-CT: Anatomy-Routed Contrastive Vision-Language
Learning for 3D Chest CT* (MICCAI 2026, TIA workshop — oral).

ARC-CT is a region-aware contrastive vision-language framework for 3D chest CT.
It learns from paired CT volumes and their radiology reports, without manual
annotations or bounding boxes, and classifies 18 abnormalities zero-shot by
comparing a volume against text prompts. It adds three things to global
image-text contrastive learning:

1. **AnatomyQFormer** — query transformer whose *anatomy* queries are masked to
   one organ each, so anatomy is injected without bounding boxes or region
   proposals; *pathology* queries attend freely; *global* queries summarise.
2. **Label-Jaccard soft InfoNCE** — the contrastive target is blended with the
   label-set overlap of each pair, so scans that share findings are no longer
   pushed apart as false negatives.
3. **Per-organ alignment** — mask-pooled organ embeddings are aligned to the
   matching report sentences.

Inference is prompt-based and needs no masks and no language model.

## Install

```bash
conda create -n arcct python=3.11 -y && conda activate arcct
pip install -r requirements.txt
```

## Data

ARC-CT trains on [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
(47,149 train / 3,039 validation volumes) and evaluates externally on the public
[RAD-ChestCT](https://zenodo.org/records/6406114) subset (3,630 volumes).

```bash
# 1. volumes -> npz (HU/1000, 1.5x1.5x3.0 mm) and TotalSegmentator masks -> 10 organ groups
python tools/prepare_data.py \
    --nifti-dir DATA/train --mask-dir DATA/train_ts --metadata DATA/train_metadata.csv \
    --out-npz DATA/npz/train --out-mask DATA/masks_192/train

# 2. 18 abnormality labels from the training reports (validation uses the official labels)
python tools/extract_labels.py --reports DATA/train_reports.csv --out DATA/train_labels.csv
```

Organ masks are produced with [TotalSegmentator](https://github.com/wasserthal/TotalSegmentator)
and reduced to ten thoracic groups: five lung lobes, trachea, heart, aorta,
mediastinal vessels, esophagus.

## Train

```bash
# Stage 1: supervised warm-start (8 epochs)
set -a; source configs/stage1.env; set +a
RAC_DATA_ROOT=DATA RAC_STAGE1_RESULTS_DIR=runs/stage1 python tools/train_stage1.py

# Stage 2: region-aware contrastive pretraining (14,400 updates, one 80 GB A100)
set -a; source configs/stage2.env; set +a
RAC_DATA_ROOT=DATA RAC_STAGE1_CKPT=runs/stage1/stage1_best.pt \
RAC_RESULTS_DIR=runs/arcct_seed0 RAC_SEED=0 python tools/train_stage2.py
```

## Evaluate

```bash
# CT-RATE validation: macro AUC, per-class AUC, precision / accuracy / F1
RAC_EVAL_CKPT=runs/arcct_seed0/CTCLIP.best.pt python tools/evaluate.py

# RAD-ChestCT external, no fine-tuning
RAC_EVAL_CKPT=runs/arcct_seed0/CTCLIP.best.pt python tools/eval_radchest.py

# Cross-modal retrieval (Table 4)
RAC_EVAL_CKPT=runs/arcct_seed0/CTCLIP.best.pt python tools/eval_retrieval.py

# Grad-CAM figure
python tools/gradcam.py --ckpt runs/arcct_seed0/CTCLIP.best.pt --volume DATA/npz/valid/<name>.npz
```

Ablations (Table 3) are single-flag overrides — see `configs/ablations.md`.
Measured numbers and the exact command for every table are in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Repository layout

```
arcct/            model and data code
  dataset.py         CT-RATE loader, HU windows, organ masks, report parsing
  image_encoder.py   3D ResNet-18 (Kinetics-initialised)
  anatomy_qformer.py AnatomyQFormer with organ-masked attention
  qformer.py         plain Q-Former (anatomy-routing ablation)
  ct_clip.py         CLIP wrapper and text transformer
  lora.py            LoRA adapters for the frozen text tower
tools/            entry points (train / evaluate / prepare data)
configs/          the exact settings behind the reported numbers
docs/             reproduction notes and the LLM prompts
```

## Pretrained weights

Checkpoints for all three seeds, plus the Stage-1 backbone, are on the Hugging Face Hub:

```bash
pip install huggingface_hub
hf download arc-ct/arc-ct --local-dir checkpoints

RAC_EVAL_CKPT=checkpoints/arcct_seed0.pt python tools/evaluate.py
```

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

## License

Code released under the MIT License (see `LICENSE`). CT-RATE and RAD-ChestCT
carry their own terms; follow those for the data.
