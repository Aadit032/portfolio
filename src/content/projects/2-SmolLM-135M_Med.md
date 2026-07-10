---
title: 'SmolLM-135M_Med'
description: 'End-to-end pipeline for continued pretraining, supervised fine-tuning, and evaluation of SmolLM-135M on medical datasets.'
pubDate: 2026-02-01
heroImage: '../../assets/smolLM-135M_med.png'
---

Medical-domain adaptation pipeline for `Aadit-032/SmolLM-135M_MedicalQA-SFT`. The project runs a two-stage training workflow — **continued pre-training (CPT)** on biomedical corpora, then **supervised fine-tuning (SFT)** on medical reasoning Q&A — with comprehensive evaluation at each stage.

---

## Overview

| Stage | Base model | Data | Goal | Output |
| --- | --- | --- | --- | --- |
| **CPT** | `HuggingFaceTB/SmolLM-135M` | PubMed, PMC, Medline, FineWeb | Domain adaptation via next-token prediction | `Aadit-032/SmolLM-135M_Med-CPT` |
| **SFT** | `Aadit-032/SmolLM-135M_Med-CPT` | medical-o1 CoT (default) or synthetic Q&A | Instruction following + medical reasoning | `Aadit-032/SmolLM-135M_MedicalQA-SFT` |
---

Both stages use **LoRA adapters** via Unsloth with 4-bit (NF4) loading and 16-bit merged export.

---

## Pipelines

### CPT pipeline (`cpt.py`)

```
Load base model → baseline evals → CPT training → merge & save → post-training evals
```

1. Load SmolLM-135M in 4-bit via Unsloth
2. Run baseline evals (perplexity, medical benchmarks, generation, lm-eval)
3. Download & tokenize 200K biomedical samples (`cpt_data.py`)
4. Train LoRA adapters for 1 epoch (`cpt_train.py`)
5. Merge and save to `SmolLM-135M_Med_Merged/`
6. Re-run all evals and compare before/after

### SFT pipeline (`sft.py`)

```
Load CPT model → pre-SFT evals → SFT training → merge & save → post-SFT evals
```

1. Load `Aadit-032/SmolLM-135M_Med-CPT` (post-CPT checkpoint)
2. Run pre-SFT evals on the CPT model
3. Load SFT dataset from `sft_data.py` (medical-o1 by default)
4. Train LoRA adapters with response-only loss masking (`sft_train.py`)
5. Merge and save to `SmolLM-135M_Med-SFT-Merged/` (also hosted on Hugging Face)
6. Re-run all evals and compare pre-SFT vs post-SFT

---

## Usage

```bash
# Install dependencies
uv sync

# --- CPT ---
uv run cpt.py              # full CPT pipeline
uv run cpt_train.py        # CPT training only
uv run cpt_data.py         # download & prepare CPT data only

# --- SFT ---
uv run sft_data.py         # build/load SFT data (defaults to medical_o1)
uv run sft.py              # full SFT pipeline
uv run sft_train.py        # SFT training only
```

### SFT data options

```bash
# Default: medical-o1 reasoning dataset (cached after first run)
uv run sft_data.py --loader medical_o1

# Synthetic: CPT split → semantic chunks → OpenRouter Q&A generation
export OPENROUTER_API_KEY="your-key"
uv run sft_data.py --loader synthetic --max-train-chunks 50

# Rebuild from scratch (ignore cached JSONL)
uv run sft_data.py --rebuild

# Choose output format: chat (default), alpaca, or raw
uv run sft_data.py --format alpaca
```

---

## Configuration

All settings live in `config.yaml`:

| Key | Value | Description |
| --- | --- | --- |
| `MODEL_NAME` | `HuggingFaceTB/SmolLM-135M` | Base model for CPT |
| `SFT_MODEL_NAME` | `Aadit-032/SmolLM-135M_Med-CPT` | Starting checkpoint for SFT |
| `SFT_LOADER` | `medical_o1` | SFT data loader (`medical_o1` or `synthetic`) |
| `SFT_TEXT_FORMAT` | `chat` | Training text format (`chat`, `alpaca`, `raw`) |
| `SFT_DATA_DIR` | `./data/sft/medical_o1` | Cached SFT dataset path |
| `SEED` | `42` | Random seed |
| `MAX_SEQ_LENGTH` | `512` | Max sequence length |
| `train_file` | `./data/train.txt` | CPT training data |
| `val_file` | `./data/val.txt` | CPT validation data |

---

## Training Details

### CPT (`cpt_train.py`)

**LoRA**

| Parameter | Value |
| --- | --- |
| Rank (`r`) | 32 |
| LoRA alpha | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `embed_tokens`, `lm_head` |
---

**Hyperparameters**

| Parameter | Value |
| --- | --- |
| Epochs | 1 |
| Per-device batch size | 32 |
| Gradient accumulation | 4 |
| Effective batch size | 128 |
| Learning rate | 2e-5 |
| Embedding LR | 2e-6 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Packing | Enabled |
---

**CPT data** (200,000 samples, 90/10 train/val split)

| Source | Samples | Field |
| --- | --- | --- |
| PubMed Abstracts | 120,000 | `abstract` |
| PMC | 40,000 | `text` |
| Medline | 20,000 | `content` |
| FineWeb | 20,000 | `text` |
---

### SFT (`sft_train.py`)

**LoRA**

| Parameter | Value |
| --- | --- |
| Rank (`r`) | 16 |
| LoRA alpha | 16 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
---

**Hyperparameters**

| Parameter | Value |
| --- | --- |
| Epochs | 1 |
| Per-device batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Learning rate | 2e-5 |
| Loss masking | Response-only (`train_on_responses_only`) |
| Packing | Enabled |
---

**SFT data loaders** (`sft_data.py`)

| Loader | Source | Description |
| --- | --- | --- |
| `medical_o1` (default) | `FreedomIntelligence/medical-o1-reasoning-SFT` | Medical CoT + answer pairs from DeepSeek-R1 |
| `synthetic` | CPT train/val split + OpenRouter API | Semantic chunks → 4 Q&A pairs per chunk (factual, analytical, synthesis, unanswerable) |
---

Both loaders append **20 general instruction pairs** (math, science, geography, refusal examples) to prevent catastrophic forgetting.

SFT data is exported in three formats: **alpaca**, **raw** (`Question:` / `Answer:`), and **chat** (ChatML-style).

---

## Evaluation

All results are saved to `./results/` as JSON.

| Eval | File suffix | What it measures |
| --- | --- | --- |
| **Perplexity** | `_untrained`, `_trained`, `_pre_sft`, `_sft` | Sliding-window PPL on PubMed Abstracts & Medline (1,000 samples each) |
| **Medical benchmarks** | same | PubMedQA (yes/no/maybe) & MedMCQA (4-option MCQ), 1,000 samples each |
| **Generation** | same | 3 general + 3 medical prompts at varying temperature/top-k |
| **LM-eval** | `results/lm_eval/*.json` | General capability: HellaSwag, PIQA, WinoGrande, ARC-Easy/Challenge, BoolQ |
---

Medical benchmarks use **single-pass log-prob scoring** — one forward pass per question by batching all answer choices together.

---

## Project Structure

```
├── cpt.py               # CPT pipeline entry point
├── cpt_train.py         # CPT LoRA training
├── cpt_data.py          # CPT dataset download & preprocessing
├── sft.py               # SFT pipeline entry point
├── sft_train.py         # SFT LoRA training
├── sft_data.py          # SFT dataset loaders (medical_o1 / synthetic)
├── model_utils.py       # Model loading (base + SFT checkpoint)
├── config.yaml          # All configuration
├── pyproject.toml       # Dependencies
├── evals/
│   ├── benchmarks.py    # PubMedQA & MedMCQA
│   ├── perplexity.py    # Sliding-window perplexity
│   ├── generation.py    # General + medical generation eval
│   └── lm_eval.py       # General capability benchmarks
├── data/
│   ├── train.txt        # CPT training data
│   ├── val.txt          # CPT validation data
│   └── sft/             # Cached SFT datasets (JSONL)
└── results/             # Evaluation outputs
```

---

## Dependencies

- Python >= 3.13
- unsloth
- datasets
- omegaconf
- evaluate
- lm-eval

Requires a CUDA GPU. For the synthetic SFT loader, set `OPENROUTER_API_KEY` in your environment.

---

## CPT Results

| Metric | Untrained | Trained | Change |
| --- | --- | --- | --- |
| PubMed PPL | 18.76 | **15.03** | **-19.9%** |
| Medline PPL | 14.24 | **11.39** | **-20.0%** |
| PubMedQA | **49.5%** | 41.5% | -8.0 pts |
| MedMCQA | 20.0% | **22.0%** | +2.0 pts |
---
CPT achieved its primary objective — domain adaptation — with ~20% perplexity reduction on biomedical text. Downstream medical QA did not improve proportionally, motivating the SFT stage on medical reasoning data.

---

## SFT Results

Evaluated on the CPT checkpoint (pre-SFT) vs the merged SFT model (post-SFT). Full outputs are in `./results/`.

**Model:** `Aadit-032/SmolLM-135M_MedicalQA-SFT`

| Metric | Pre-SFT (CPT) | Post-SFT | Change |
| --- | --- | --- | --- |
| PubMed PPL | 17.34 | **17.14** | **-1.2%** |
| Medline PPL | 13.05 | **12.90** | **-1.2%** |
| PubMedQA | 45.1% | **48.9%** | **+3.8 pts** |
| MedMCQA | 24.2% | 24.1% | -0.1 pts |
---

SFT recovered PubMedQA accuracy toward the untrained baseline (49.5%) while keeping biomedical perplexity stable. MedMCQA was essentially unchanged — a harder 4-option MCQ benchmark that likely needs more targeted training data or longer fine-tuning.

**General capability (lm-eval, post-SFT)**

| Task | Accuracy |
| --- | --- |
| HellaSwag | 34.5% |
| PIQA | 68.2% |
| WinoGrande | 51.6% |
| ARC-Easy | 60.1% |
| ARC-Challenge | 25.6% |
| BoolQ | 59.8% |
---

Generation samples (`generation_pre_sft.json` vs `generation_sft.json`) show modest gains in instruction-following structure, but outputs remain repetitive at 135M scale — expected for a model this size without RLHF or larger SFT corpora.