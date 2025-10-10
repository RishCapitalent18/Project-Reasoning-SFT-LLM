# Reproducibility Guide

## 1) ARC Environment Setup
```bash
module load Miniconda3
module load CUDA/12.6.0

# Create and activate env
conda create -n myenv python=3.10 -y
conda activate myenv

# LLaMA-Factory (if using from source)
# cd LLaMA-Factory && pip install -e ".[torch,metrics]" --no-build-isolation

# lighteval
pip install lighteval[vllm,extended_tasks,math,dev]
# If using a local clone, also: pip install -e lighteval/

# Optional accelerators
pip install flash-attn --no-build-isolation
pip install deepspeed==0.16.8
```

Login to Hugging Face:
```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 2) Datasets
You will need two JSON files (or corresponding Hugging Face dataset repos):
- Random 15k: train_subset_15k.json
- LIMOPro 15k: acereason_limopro_15k.json

Place them under LLaMA-Factory/data (or host on Hugging Face).

Update LLaMA-Factory `data/dataset_info.json` with entries similar to:
```json
{
  "train_subset_15k": { "file_name": "train_subset_15k.json" },
  "acereason_limopro_15k": { "file_name": "acereason_limopro_15k.json" }
}
```
A patch file is provided at submission/data/dataset_info_patch.json.

## 3) Baseline Evaluation
From the repo root:
```bash
bash submission/scripts/baseline.sh
```
Outputs (metrics + details) are saved under:
- submission/results/raw/

## 4) Train SFT (Random 15k)
```bash
llamafactory-cli train submission/configs/sft_config.yaml
```
Adjust `output_dir` and `dataset` in the YAML if needed.

## 5) Train SFT (LIMOPro 15k)
```bash
llamafactory-cli train submission/configs/sft_limopro.yaml
```
Adjust `output_dir` and `dataset` in the YAML if needed.

## 6) Evaluate Fine-tuned Models
Random SFT:
```bash
bash submission/scripts/eval_sft.sh
```

LIMOPro SFT:
```bash
bash submission/scripts/eval_limopro.sh
```

## 7) Expected Outputs
- Raw lighteval logs: submission/results/raw/
- Summary tables:
  - submission/results/summary.md
  - submission/results/summary.csv

## Notes
- SLURM directives in scripts may need to be adapted to your ARC partition/account.
- If using different GPU types or memory, adjust vLLM parameters (gpu_memory_utilization, tensor_parallel_size, etc.).
- Ensure `template: qwen` is used consistently in training and inference with Qwen models.
