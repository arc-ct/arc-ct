# LLM prompts

Two offline LLM passes are used. Neither runs at training or inference time.

## 1. Abnormality label extraction (training reports)

Following the MPS-CT protocol, the 18 CT-RATE abnormality labels are extracted
from each *training* report. The CT-RATE validation labels are the official
human-verified ones and are never model-generated.

See `tools/extract_labels.py` for the exact prompt and the parsing of the
model's response into an 18-way binary vector.

## 2. Per-organ sentence extraction (per-organ alignment)

Each report is split into the sentences describing each of the ten organ groups,
which supplies the text side of the per-organ alignment loss.

See `tools/extract_organ_sentences.py` for the prompt. The loader falls back to
rule-based keyword matching (`localize_findings` in `arcct/dataset.py`) when no
cache is supplied, so the pipeline runs without any LLM access.
