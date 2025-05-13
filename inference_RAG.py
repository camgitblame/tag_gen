import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import wandb
from sentence_transformers import SentenceTransformer
import faiss
import json
wandb.init(project="tag_gen", name="inference-RAG")


# 📍 Paths
MODEL_DIR = "gpt2-RAG-output"
TOKENIZER_DIR = "tokenizer"
EVAL_FILE = "eval.csv"

# Load tokenizer and model
print("🚀 Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 📦 Load retrieval model and FAISS index
print("📡 Loading retriever...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("faiss_overview.index")
with open("id_to_text.json", "r") as f:
    all_overviews = json.load(f)


# 🧪 Load eval data
print("📂 Loading evaluation set...")
df = pd.read_csv(EVAL_FILE)
df = df[df["overview"].notna() & df["tagline"].notna()].head(100)  # Optional cap

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
    # 🔍 Retrieve similar overviews
    query_vec = encoder.encode([overview])
    D, I = faiss_index.search(query_vec, k=3)  # top-3, adjust if needed
    retrieved = [all_overviews[i] for i in I[0] if all_overviews[i] != overview][:2]

    # 🧠 Format prompt like training
    retrieved_block = "\n".join([f"Retrieved Overview {j+1}: {txt}" for j, txt in enumerate(retrieved)])
    input_text = f"{retrieved_block}\nOriginal Overview: {overview}\nTagline:"


    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            # Increase the max_new_tokens to allow for longer taglines
            max_new_tokens=32,  
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=1.0
        )

        


    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_tagline = decoded.split("Tagline:")[-1].strip().split("\n")[0].strip()
    # Remove "Overview" artifacts
    for artifact in ["Overview:", "overview", "OVERVIEW"]:
        if generated_tagline.lower().startswith(artifact.lower()):
            generated_tagline = generated_tagline[len(artifact):].strip()

    # Added sentence cutoff cleanup
    import re
    match = re.search(r"(.+?[.!?])\s", generated_tagline)
    if match:
        generated_tagline = match.group(1).strip()

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
output_df.to_csv("generated_vs_original_RAG.csv", index=False)
print("✅ Output saved to generated_vs_original_RAG.csv")

