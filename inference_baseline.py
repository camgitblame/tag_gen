import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import wandb
import re
wandb.init(project="tag_gen", name="inference-baseline")


# 📍 Paths
MODEL_DIR = "gpt2-output"
TOKENIZER_DIR = "tokenizer"
EVAL_FILE = "eval.csv"

# Load tokenizer and model
print("🚀 Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# 🧪 Load eval data
print("📂 Loading evaluation set...")
df = pd.read_csv(EVAL_FILE)
df = df[df["overview"].notna() & df["tagline"].notna()]

# 📊 Init metrics
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
generated_list = []
reference_list = []

# 🌀 Loop over samples
print("🔎 Running inference...")
for _, row in df.iterrows():
    title = row["title"]
    overview = row["overview"]
    reference = row["tagline"]
    input_text = f"Overview: {overview}\nTagline:"

    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=16,  
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,         # Beam width
            early_stopping=True, # Stop at eos
            no_repeat_ngram_size=2,  # Discourage repeats
            do_sample=False      
        )


    # Decode the output
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_tagline = decoded.split("Tagline:")[-1].strip().split("\n")[0].strip()
    # Remove "Overview" artifacts
    for artifact in ["Overview:", "overview", "OVERVIEW"]:
        if generated_tagline.lower().startswith(artifact.lower()):
            generated_tagline = generated_tagline[len(artifact):].strip()

    # Added sentence cutoff cleanup
    match = re.search(r"(.+?[.!?])\s", generated_tagline)
    if match:
        generated_tagline = match.group(1).strip()

    # Remove stray trailing quotation mark
    generated_tagline = generated_tagline.strip('"')
    generated_list.append(generated_tagline)
    reference_list.append(reference)

# 📏 Compute ROUGE
print("📊 Evaluating ROUGE...")
rouge1_scores, rougeL_scores = [], []
for ref, hyp in zip(reference_list, generated_list):
    scores = rouge.score(ref, hyp)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

# 📐 Compute BERTScore
print("📊 Evaluating BERTScore...")
P, R, F1 = bert_score(generated_list, reference_list, lang="en", verbose=True)

# 📈 Summary
print("\n📈 Evaluation Summary:")
print(f"Avg ROUGE-1 F1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"Avg ROUGE-L F1: {sum(rougeL_scores)/len(rougeL_scores):.4f}")
print(f"Avg BERTScore F1: {F1.mean().item():.4f}")


wandb.log({
    "avg_rouge1_f1": sum(rouge1_scores)/len(rouge1_scores),
    "avg_rougeL_f1": sum(rougeL_scores)/len(rougeL_scores),
    "avg_bertscore_f1": F1.mean().item()
})

# 💾 Save to CSV
print("📁 Saving outputs to CSV...")
output_df = pd.DataFrame({
    "Title": df["title"].tolist(),   
    "Original": reference_list,
    "Generated": generated_list
})
output_df.to_csv("generated_vs_original_with_beam.csv", index=False)
print("✅ Output saved to generated_vs_original_with_beam.csv")

