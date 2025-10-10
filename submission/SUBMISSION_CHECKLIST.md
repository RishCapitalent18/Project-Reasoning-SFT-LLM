# Submission Checklist

Use this checklist to verify the package meets the submission requirements.

Core deliverables:
- [x] submission/README.md (package index and contents)
- [x] submission/REPORT.md (methods, hyperparameters, results analysis)
- [x] submission/HOWTO.md (reproducibility guide)
- [x] submission/HUGGINGFACE.md (publishing guide)
- [x] submission/SUBMISSION_CHECKLIST.md (this file)

Results:
- [x] Raw outputs included under submission/results/raw/
  - [x] resultsStep2.1.out (baseline: AIME24/25, MATH-500, GPQA, LCB)
  - [x] resultsStep2.2.out (baseline: MMLU-Redux-2)
  - [x] results6.out (SFT Random)
  - [x] results7.out (SFT LIMOPro)
- [x] Summary tables
  - [x] submission/results/summary.md
  - [x] submission/results/summary.csv

Configs:
- [x] submission/configs/sft_config.yaml (Random 15k SFT)
- [x] submission/configs/sft_limopro.yaml (LIMOPro 15k SFT)
- [x] submission/configs/eval_sft.yaml (Evaluation config; outputs to submission/results/raw)

Scripts (paths normalized):
- [x] submission/scripts/baseline.sh
- [x] submission/scripts/eval_sft.sh
- [x] submission/scripts/eval_limopro.sh

Data docs:
- [x] submission/data/dataset_info_patch.json (example entries to add)
- [x] submission/data/NOTE.md (where to place/get datasets)

Content checks:
- [x] REPORT.md contains all six benchmarks and correct metrics:
  - [x] AIME24 pass@k_with_k
  - [x] AIME25 pass@k_with_k&n
  - [x] MATH-500 pass@k_with_k&n
  - [x] GPQA-Diamond gpqa_pass@k_with_k
  - [x] LiveCodeBench codegen_pass@1:16
  - [x] MMLU-Redux-2 acc
- [x] Numbers match raw files:
  - Baseline: AIME24 0.1000; AIME25 0.0000; MATH-500 0.6340; GPQA 0.2929; LCB 0.0858; MMLU acc 0.6393
  - SFT Random: AIME24 0.0333; AIME25 0.0333; MATH-500 0.5060; GPQA 0.3081; LCB 0.0485; MMLU acc 0.2356
  - SFT LIMOPro: AIME24 0.0000; AIME25 0.1000; MATH-500 0.5600; GPQA 0.2727; LCB 0.0448; MMLU acc 0.4404

Notes for graders/reruns:
- SLURM partition/account settings in scripts may require edits for your cluster.
- MODEL paths in eval scripts may need to be adjusted to your actual trained checkpoints.
- Ensure LLaMA-Factory’s dataset_info.json contains the entries in submission/data/dataset_info_patch.json (or use HF-hosted datasets).
- vLLM generation parameters (max_new_tokens, temperature, top_p) are set for parity with the provided runs; adjust only if needed.
