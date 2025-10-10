# Report: SFT Training LLM for Improved Reasoning

## Abstract
We fine-tuned Qwen/Qwen2.5-3B-Instruct on two 15k subsets of AceReason-1.1-SFT: a random sample and a LIMOPro-selected sample. We evaluated baseline and both fine-tuned models on AIME24/25, MATH-500, GPQA-Diamond, LiveCodeBench codegeneration, and MMLU-Redux-2.

## Methods
- Model: Qwen/Qwen2.5-3B-Instruct
- Framework: LLaMA-Factory
- Finetuning type: Full
- Deepspeed: ds_z3_offload
- Flash Attention: fa2
- Template: qwen
- Cutoff length: 16384 (random SFT), 8192 (LIMOPro SFT)
- Mixed precision: bf16

### Data Selection
- Random 15k: uniform random sample from AceReason-1.1-SFT 100k subset.
- LIMOPro 15k: curated subset using LIMOPro pruning/selection strategy to favor more informative prompt-response pairs for the base model.

### Training Hyperparameters (random SFT)
From configs/sft_config.yaml:
- per_device_train_batch_size: 1
- gradient_accumulation_steps: 8
- learning_rate: 1e-6
- num_train_epochs: 2
- lr_scheduler_type: cosine
- warmup_ratio: 0.05

### Training Hyperparameters (LIMOPro SFT)
From configs/sft_limopro.yaml:
- per_device_train_batch_size: 1
- gradient_accumulation_steps: 16
- learning_rate: 5e-6
- num_train_epochs: 3
- lr_scheduler_type: cosine
- warmup_ratio: 0.05
- gradient_checkpointing: true

## Results
Benchmarks and metrics:

- AIME24 (pass@k_with_k):
  - Baseline: 0.1000
  - SFT-random: 0.0333
  - SFT-LIMOPro: 0.0000

- AIME25 (pass@k_with_k&n):
  - Baseline: 0.0000
  - SFT-random: 0.0333
  - SFT-LIMOPro: 0.1000

- MATH-500 (pass@k_with_k&n):
  - Baseline: 0.6340
  - SFT-random: 0.5060
  - SFT-LIMOPro: 0.5600

- GPQA-Diamond (gpqa_pass@k_with_k):
  - Baseline: 0.2929
  - SFT-random: 0.3081
  - SFT-LIMOPro: 0.2727

- LiveCodeBench Codegeneration (codegen_pass@1:16):
  - Baseline: 0.0858
  - SFT-random: 0.0485
  - SFT-LIMOPro: 0.0448

- MMLU-Redux-2 (acc):
  - Baseline: 0.6393
  - SFT-random: 0.2356
  - SFT-LIMOPro: 0.4404

### Discussion
- Random SFT exhibits degradation on several general benchmarks (notably MMLU acc: 0.2356 vs 0.6393 baseline) and mixed effects on reasoning/math tasks.
- LIMOPro SFT recovers a substantial portion of MMLU (0.4404 vs 0.2356 random), improves AIME25 and MATH-500 over random, but still trails baseline on MMLU and codegen.
- These signals suggest catastrophic forgetting and domain shift induced by SFT on reasoning-heavy math data; selection strategies like LIMOPro help mitigate forgetting compared to random sampling.

### Step 7: Advanced Data Selection (LIMOPro)

This corresponds to README Step 7 (advanced data selection). We used a curated 15k subset via LIMOPro and observed the following changes versus the Random 15k SFT:
- AIME24 (pass@k_with_k): 0.0333 → 0.0000 (−0.0333)
- AIME25 (pass@k_with_k&n): 0.0333 → 0.1000 (+0.0667)
- MATH-500 (pass@k_with_k&n): 0.5060 → 0.5600 (+0.0540)
- GPQA-Diamond (gpqa_pass@k_with_k): 0.3081 → 0.2727 (−0.0354)
- LiveCodeBench (codegen_pass@1:16): 0.0485 → 0.0448 (−0.0037)
- MMLU-Redux-2 (acc): 0.2356 → 0.4404 (+0.2048)

Summary: LIMOPro substantially recovers general knowledge (MMLU) and improves some reasoning tasks (e.g., AIME25, MATH-500) relative to random selection, with small trade-offs on GPQA/codegen and AIME24. Raw logs for Step 7 are included:
- submission/results/raw/results7.out
- submission/results/raw/eval_limopro_92004 (step 7).out

See also:
- Config: submission/configs/sft_limopro.yaml
- Eval script: submission/scripts/eval_limopro.sh

### Compute & Environment
- ARC (Miniconda3, CUDA 12.6)
- Tooling: LLaMA-Factory + lighteval + vLLM worker

## Links (placeholders)
- Hugging Face Model(s): see HUGGINGFACE.md
- Hugging Face Dataset(s): see HUGGINGFACE.md

## Appendices
- Full per-subject MMLU breakdowns are available in results6.out and results7.out for SFT runs, and resultsStep2.2.out for baseline.
