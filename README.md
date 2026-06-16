# CSE4078S26_Grp5 — Code Repository

**Course:** CSE4078 Spring 2026  
**Group:** 5  
**University:** Marmara University, Department of Computer Engineering  
**Project:** Fine-Tuning Small Open-Source LLMs for Turkish Medical QA

---

## Group Members

| Name | Student ID |
|------|-----------|
| Yiğitcan Değişme | 150121062 |
| Emir Devran | 150119749 |
| Kerem Kolay | 150119773 |
| Yunus Emre Gül | 150122061 |
| Ömer Faruk Sevinç | 150120050 |

---

## Repository Structure

```
CSE4078S26_Grp5_Code/
│
├── README.md                          # This file
│
├── 01_Deadline1_Baseline.ipynb        # Deadline 1: Data prep + baseline evaluation
│                                      #   - Dataset filtering & deduplication
│                                      #   - Stratified train/test split (seed=42)
│                                      #   - Baseline inference: Qwen3-4B & Phi-3.5-mini
│                                      #   - Token F1, Exact Match, MiniLM, BERTScore, ROUGE
│
├── 02_Deadline2_SFT_v1.ipynb          # Deadline 2 (SFT-v1): Conservative fine-tuning
│                                      #   - QLoRA rank-16, 15,000 examples, 1 epoch
│                                      #   - Unsloth + TRL SFTTrainer
│                                      #   - Post-SFT evaluation on benchmark_test.jsonl
│
└── 03_Deadline2_SFT_v2_Optimized.ipynb # Deadline 2 (SFT-v2): Optimized fine-tuning
                                        #   - QLoRA rank-32, 39,612 examples, 3 epochs
                                        #   - Cosine LR schedule, AdamW 8-bit
                                        #   - Full evaluation: 8 metrics + field-level
```

---

## Dataset

**Primary dataset:** `alibayram/doktorsitesi`  
Available at: https://huggingface.co/datasets/alibayram/doktorsitesi

**9 medical specialties used:**
- beyin-ve-sinir-cerrahisi
- uroloji
- ortopedi-ve-travmatoloji
- dahiliye-ve-ic-hastaliklari
- genel-cerrahi
- kulak-burun-bogaz-hastaliklari
- fiziksel-tip-ve-rehabilitasyon
- kardiyoloji
- kalp-damar-cerrahisi

**Splits (seed=42, stratified):**
| Split | File | Rows |
|-------|------|------|
| SFT training pool | sft_train.jsonl | 39,612 |
| SFT-v1 subsample | — | 15,000 |
| Benchmark test | benchmark_test.jsonl | 1,500 |
| Train–test overlap | — | **0** |

> ⚠️ The benchmark test set (`benchmark_test.jsonl`) was **never used during training** at any stage.

---

## Models

| Model | Parameters | Role |
|-------|-----------|------|
| Qwen/Qwen3-4B | ~4.0B | Baseline + Fine-tuned |
| microsoft/Phi-3.5-mini-instruct | ~3.8B | Baseline only |

Both require a Hugging Face token (`HF_TOKEN`) set as a Colab secret.

---

## How to Run

### Prerequisites
All notebooks run on **Google Colab Pro** with an **NVIDIA A100 40GB GPU**.

```bash
# Packages installed inside notebooks:
pip install datasets transformers accelerate bitsandbytes huggingface_hub tqdm
pip install sentence-transformers bert-score rouge-score
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps peft trl
```

### Step 1 — Data Preparation & Baseline Evaluation
Open `01_Deadline1_Baseline.ipynb` in Colab:
1. Set `HF_TOKEN` in Colab Secrets
2. Run **Section 1** — Install dependencies
3. Run **Section 2** — HF login
4. Run **Section 3** — Dataset filtering, deduplication, stratified split
5. Run **Section 4** — Baseline inference (Qwen3-4B + Phi-3.5-mini)
6. Run **Section 5** — Compute all 8 metrics (Token F1, ROUGE, MiniLM, BERTScore)

### Step 2 — SFT-v1 (Conservative Fine-Tuning)
Open `02_Deadline2_SFT_v1.ipynb` in Colab:
1. Run **Section 1** — Model setup (Qwen3-4B + QLoRA rank-16)
2. Run **Section 2** — SFT data preparation (15,000 stratified examples)
3. Run **Section 3** — QLoRA fine-tuning (1 epoch, linear LR)
4. Run **Section 4** — Save adapter
5. Run **Section 5** — Post-SFT inference on benchmark_test.jsonl
6. Run **Section 6** — Evaluation (8 metrics)

### Step 3 — SFT-v2 (Optimized Fine-Tuning)
Open `03_Deadline2_SFT_v2_Optimized.ipynb` in Colab:
1. Run **Section 1** — Model setup (Qwen3-4B + QLoRA rank-32)
2. Run **Section 2** — SFT data preparation (full 39,612 examples)
3. Run **Section 3** — QLoRA fine-tuning (3 epochs, cosine LR)
4. Run **Section 4** — Save adapter
5. Run **Section 5** — Post-SFT inference on benchmark_test.jsonl
6. Run **Section 6** — Evaluation (8 metrics + field-level Token F1)

---

## Results Summary

| Configuration | Token F1 | ROUGE-L | MiniLM Sim | BERTScore F1 | Avg Tokens |
|--------------|----------|---------|------------|--------------|------------|
| Phi-3.5-mini (baseline) | 0.0542 | 0.0769 | 0.3424 | 0.8183 | 26.6 |
| Qwen3-4B (baseline) | 0.0653 | 0.0757 | 0.3798 | 0.8191 | 36.4 |
| Qwen3-4B SFT-v1 | 0.0811 | 0.0978 | 0.4121 | 0.8283 | 31.6 |
| **Qwen3-4B SFT-v2** | **0.1006** | **0.1183** | **0.4504** | **0.8363** | **25.7** |

---

## Hyperparameters

| Parameter | SFT-v1 | SFT-v2 |
|-----------|--------|--------|
| LoRA rank (r) | 16 | 32 |
| LoRA alpha | 16 | 64 |
| LoRA dropout | 0.0 | 0.05 |
| Training examples | 15,000 | 39,612 |
| Epochs | 1 | 3 |
| Learning rate | 2×10⁻⁴ | 1×10⁻⁴ |
| LR scheduler | Linear | Cosine |
| Effective batch size | 8 | 16 |
| Max seq length | 1,024 | 2,048 |
| Optimizer | AdamW 8-bit | AdamW 8-bit |
| GPU | A100 40GB | A100 40GB |

---

## Reproducibility

- All random seeds: `seed=42` (data splits), `seed=3407` (training)
- Train–test overlap: **0 records** (verified)
- Benchmark test set was **never used during training**
- All results are reproducible by running notebooks in order

---

## Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Unsloth | Latest (GitHub) | Fast QLoRA training |
| TRL (SFTTrainer) | Latest | Supervised fine-tuning |
| PEFT | Latest | LoRA adapter management |
| BitsAndBytes | Latest | 4-bit NF4 quantization |
| sentencetransformers | Latest | MiniLM semantic similarity |
| bert-score | Latest | BERTScore F1 |
| rouge-score | Latest | ROUGE-1/2/L |
| datasets | Latest | HuggingFace dataset loading |
