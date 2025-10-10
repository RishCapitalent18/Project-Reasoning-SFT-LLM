
import json, torch, numpy as np, pandas as pd, re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


SOURCE_PATH = "/home/apoorvachavali/A-TML/LLaMA-Factory/data/acereason11_100k.json"
OUTPUT_PATH = "/home/apoorvachavali/A-TML/LLaMA-Factory/data/acereason_limopro_15k.json"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TARGET_SAMPLES = 15000
N_CLUSTERS = 50
DEVICE = "cuda"
BATCH_SIZE = 8  # For faster generation

print("="*90)
print(" LIMOPro-light: Model-Aware Data Selection (OPTIMIZED)")
print("="*90)


df = pd.read_json(SOURCE_PATH)
print(f"✓ Loaded {len(df):,} samples\n")


def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed{([^}]+)}', text)
    return matches[-1] if matches else None

def compute_difficulty(row):
    out = row["output"]
    response_len = len(out.split())
    reasoning_terms = ["step", "then", "thus", "finally", "therefore", "hence", "so"]
    reasoning_score = sum(out.lower().count(w) for w in reasoning_terms)
    math_symbols = ["\\frac", "\\sqrt", "\\sum", "^", "_", "=", "\\int", "\\prod"]
    math_complexity = sum(out.count(s) for s in math_symbols)
    has_ans = 1 if extract_boxed_answer(out) else 0
    return (
        min(response_len/1200,1)*0.3 +
        min(reasoning_score/15,1)*0.3 +
        min(math_complexity/30,1)*0.25 +
        has_ans*0.15
    )

print("Computing difficulty scores...")
df["difficulty"] = df.apply(compute_difficulty, axis=1)
df["difficulty_norm"] = (df["difficulty"] - df["difficulty"].min()) / (df["difficulty"].max() - df["difficulty"].min() + 1e-9)
print(f"Mean difficulty: {df['difficulty'].mean():.3f}\n")


print("Computing model familiarity scores...")
print("      Loading sentence transformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# OPTION A: Use embedding similarity (NO generation needed - FAST!)
print("      Encoding instructions and outputs...")
instruction_embs = embedder.encode(df["instruction"].tolist(), batch_size=128, show_progress_bar=True)
output_embs = embedder.encode(df["output"].tolist(), batch_size=128, show_progress_bar=True)

# Compute similarity between instruction and its output
# High similarity = simple/formulaic, Low similarity = complex reasoning
df["instr_out_sim"] = [
    cosine_similarity([instruction_embs[i]], [output_embs[i]])[0][0] 
    for i in range(len(df))
]

# OPTION B: Sample-based model generation (for validation)
print("      Generating model responses for subset (validation)...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

def batch_generate(instructions, batch_size=BATCH_SIZE):
    """Fast batch generation"""
    results = []
    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
        batch = instructions[i:i+batch_size]
        
        # Format prompts
        prompts = [f"Q: {instr}\nA:" for instr in batch]
        
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tok.eos_token_id
            )
        
        batch_results = [tok.decode(o[inputs['input_ids'].shape[1]:], skip_special_tokens=True) for o in outputs]
        results.extend(batch_results)
    
    return results

# Generate for 1000 samples (validation set)
val_size = 1000
df_val = df.sample(val_size, random_state=42)
val_instructions = df_val["instruction"].tolist()
val_gold = df_val["output"].tolist()

generated = batch_generate(val_instructions)

# Compute similarity between generated and gold
val_sims = []
for gen, gold in tqdm(zip(generated, val_gold), total=len(generated), desc="Computing similarities"):
    emb_gen = embedder.encode(gen)
    emb_gold = embedder.encode(gold)
    sim = cosine_similarity([emb_gen], [emb_gold])[0][0]
    val_sims.append(sim)

df_val["model_sim"] = val_sims

print(f" Model similarity (validation): {np.mean(val_sims):.3f} ± {np.std(val_sims):.3f}")

# Free up GPU memory
del model
torch.cuda.empty_cache()
print(" Model unloaded\n")

# Combine both similarity signals
# Use instruction-output similarity as proxy, calibrated by model validation
calibration_factor = np.mean(val_sims) / df_val["instr_out_sim"].mean()
df["model_familiarity"] = df["instr_out_sim"] * calibration_factor
df["model_familiarity"] = np.clip(df["model_familiarity"], 0, 1)

print(f" Mean model familiarity: {df['model_familiarity'].mean():.3f}\n")


print(" Computing value scores...")

# Higher value = good difficulty + low familiarity (model struggles)
df["value_score"] = (
    0.5 * df["difficulty_norm"] +           # Want challenging problems
    0.5 * (1 - df["model_familiarity"])     # Want unfamiliar problems
)

print(f"      ✓ Mean value score: {df['value_score'].mean():.3f}")
print(f"      ✓ Top 10% value score threshold: {df['value_score'].quantile(0.9):.3f}\n")



print("Clustering for diversity...")
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42, max_iter=300)
df["cluster"] = kmeans.fit_predict(instruction_embs)
print(f" Clustered into {N_CLUSTERS} groups\n")


print("Stratified sampling by value score...")

samples_per_cluster = TARGET_SAMPLES // N_CLUSTERS
remainder = TARGET_SAMPLES % N_CLUSTERS

selected = []
rng = np.random.default_rng(42)

for cid in range(N_CLUSTERS):
    cdata = df[df["cluster"] == cid]
    if len(cdata) == 0:
        continue
    
    n = samples_per_cluster + (1 if cid < remainder else 0)
    n = min(n, len(cdata))
    
    # Weight by value score (higher = more likely to select)
    weights = cdata["value_score"].values
    weights = weights / weights.sum()
    
    idx = rng.choice(cdata.index, size=n, replace=False, p=weights)
    selected.append(df.loc[idx])

df_final = pd.concat(selected, ignore_index=True)

# Ensure exactly TARGET_SAMPLES
if len(df_final) > TARGET_SAMPLES:
    df_final = df_final.sample(n=TARGET_SAMPLES, random_state=42)
elif len(df_final) < TARGET_SAMPLES:
    needed = TARGET_SAMPLES - len(df_final)
    remaining = df[~df.index.isin(df_final.index)]
    extra = remaining.nlargest(needed, 'value_score')
    df_final = pd.concat([df_final, extra], ignore_index=True)

assert len(df_final) == TARGET_SAMPLES

print(f" Final selection: {len(df_final):,} samples")
print(f" Mean value score: {df_final['value_score'].mean():.3f}")
print(f" Mean difficulty: {df_final['difficulty'].mean():.3f}")
print(f" Mean familiarity: {df_final['model_familiarity'].mean():.3f}\n")


records = []
for _, r in df_final.iterrows():
    records.append({
        "instruction": r["instruction"],
        "input": r.get("input", ""),
        "output": r["output"],
        "system": "Please reason step by step, and put your final answer within \\boxed{}."
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(records, f, indent=2)

# Save statistics
stats = {
    "method": "LIMOPro-light",
    "total_samples": len(records),
    "value_score_mean": float(df_final["value_score"].mean()),
    "difficulty_mean": float(df_final["difficulty"].mean()),
    "familiarity_mean": float(df_final["model_familiarity"].mean()),
    "clusters": N_CLUSTERS
}

with open(OUTPUT_PATH.replace('.json', '_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print("="*90)
print(f" SUCCESS! Saved {len(records):,} model-aware examples to:")
print(f"   {OUTPUT_PATH}")
print(f"\n Selection favored:")
print(f"   • High difficulty: {df_final['difficulty'].mean():.3f}")
print(f"   • Low familiarity: {df_final['model_familiarity'].mean():.3f} (model struggles)")
print(f"   • Diverse coverage: {N_CLUSTERS} clusters")
print("="*90)