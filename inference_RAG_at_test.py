import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import wandb
from sentence_transformers import SentenceTransformer
import faiss
import json
import re
import random
wandb.init(project="tag_gen", name="inference-RAG-infer-only")


# 📍 Paths
MODEL_DIR = "gpt2-output"
TOKENIZER_DIR = "tokenizer"
EVAL_FILE = "eval.csv"

def truncate_text(text, max_words=50):
    return " ".join(text.split()[:max_words])

FEW_SHOT_EXAMPLES = [
    ("A robot is sent back in time to protect a child.", "He'll be back."),
    ("A magical nanny reunites a family.", "Practically perfect in every way."),
    ("Two lovers meet aboard a doomed ocean liner.", "Nothing on Earth could come between them."),
    ("A team of superheroes battles an alien invasion.", "Avengers assemble."),
    ("A shark terrorizes a beach town.", "Don't go in the water.")
]

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

    # Retrieve similar overviews
    query_vec = encoder.encode([overview])
    D, I = faiss_index.search(query_vec, k=3)  # top-3, adjust if needed
    retrieved = [truncate_text(all_overviews[i]) for i in I[0] if all_overviews[i] != overview][:2]
    retrieved_block = "\n".join(retrieved)


    # Add few-shot in-context examples
    sampled_examples = random.sample(FEW_SHOT_EXAMPLES, 2)
    example_block = ""
    for ex_ov, ex_tag in sampled_examples:
        example_block += f"Overview: {ex_ov}\nTagline: {ex_tag}\n\n"

    # Combine few-shot examples + current input
    input_text = f"{example_block}{retrieved_block}\n\nOverview: {overview}\nTagline:"

    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            # Increase max_new_tokens to 32
            max_new_tokens=32,  
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,
            early_stopping=True,
            # Increase no_repeat_ngram_size to 3 to reduce repetition
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=1.0
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
output_df.to_csv("generated_vs_original_RAG_infer.csv", index=False)
print("✅ Output saved to generated_vs_original_RAG_infer.csv")

