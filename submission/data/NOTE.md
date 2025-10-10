# Dataset Notes

Large dataset JSON files are not committed to the repository. To reproduce the training runs you will need the following datasets:

- Random 15k subset: train_subset_15k.json
- LIMOPro 15k subset: acereason_limopro_15k.json

Options to provide datasets:
1) Local files:
   - Place JSONs in the LLaMA-Factory data directory (e.g., LLaMA-Factory/data/).
   - Update LLaMA-Factory/data/dataset_info.json with entries similar to the patch in submission/data/dataset_info_patch.json.

2) Hugging Face datasets:
   - Host the datasets on Hugging Face (recommended).
   - In dataset_info.json, use hf_hub parameters per LLaMA-Factory’s format to point to the hosted dataset repos.

Checklist:
- [ ] Ensure dataset_info.json contains:
  {
    "train_subset_15k": { "file_name": "train_subset_15k.json" },
    "acereason_limopro_15k": { "file_name": "acereason_limopro_15k.json" }
  }
- [ ] Verify the “template: qwen” setting in training configs matches your data formatting.
- [ ] Confirm data paths are accessible from the machine/cluster where you run training.
