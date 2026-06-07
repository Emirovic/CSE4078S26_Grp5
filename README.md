# CSE4078S26_Grp5

**CSE4078 Spring 2026 — Introduction to NLP — Term Project**  
**Group 5 — Marmara University, Department of Computer Engineering**

| Member | Student ID |
|--------|-----------|
| Yiğitcan Değişme | 150121062 |
| Emir Devran | 150119749 |
| Kerem Kolay | 150119773 |
| Yunus Emre Gül | 150122061 |
| Ömer Faruk Sevinç | 150120050 |

---

## Project Summary

We evaluate two small open-source LLMs on a Turkish medical QA benchmark, then apply QLoRA fine-tuning to the stronger candidate.

- **Dataset:** alibayram/doktorsitesi (167,732 Turkish medical QA pairs)
- **Models evaluated:** Qwen/Qwen3-4B, microsoft/Phi-3.5-mini-instruct
- **Fine-tuned model:** Qwen/Qwen3-4B with QLoRA (LoRA rank 16, 4-bit NF4)
- **Training:** 15,000 stratified examples, 1 epoch, Google Colab Pro A100 40GB
- **Benchmark:** 1,500 held-out examples, 8 evaluation metrics

---

## Results Summary

| Metric | Qwen3-4B Baseline | Post-SFT | Delta |
|--------|-------------------|----------|-------|
| Token F1 | 0.0653 | 0.0811 | +0.0158 |
| ROUGE-1 F1 | 0.1109 | 0.1302 | +0.0193 |
| ROUGE-L F1 | 0.0757 | 0.0978 | +0.0221 |
| MiniLM Sim. | 0.3798 | 0.4121 | +0.0323 |
| BERTScore F1 | 0.8191 | 0.8283 | +0.0092 |
| Avg Tokens | 36.4 | 31.6 | −4.7 |

---

## Repository Structure

- NLP_PROJE.ipynb — Deadline 1: Baseline evaluation (Qwen3-4B and Phi-3.5-mini)
- CSE4078S26_Grp5_Deadline2.ipynb — Deadline 2: Fine-tuning, inference, evaluation
- README.md

---

## How to Reproduce

**Install dependencies:**

    pip install datasets pandas scikit-learn huggingface_hub
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate transformers
    pip install rouge-score bert-score sentence-transformers

**Steps:**

1. Run Section 1 of CSE4078S26_Grp5_Deadline2.ipynb — Data preprocessing. Requires HuggingFace token for alibayram/doktorsitesi. Outputs: sft_train.jsonl (39,612 rows), benchmark_test.jsonl (1,500 rows)

2. Run Sections 2–5 — QLoRA fine-tuning on Google Colab Pro A100 40GB. Output: qwen3_doc_lora_15k/ adapter weights

3. Run Section 6 — Inference on benchmark_test.jsonl. Output: qwen3_finetuned_predictions.csv

4. Run Section 7 — Compute all 8 metrics. Output: qwen3_finetuned_metrics.csv

---

## Key Settings

| Setting | Value |
|---------|-------|
| Random seed | 42 |
| max_new_tokens | 96 |
| Decoding | Greedy (do_sample=False) |
| Qwen thinking mode | Disabled (enable_thinking=False) |
| Train-test overlap | 0 (verified) |
