import torch
import pandas as pd
import ast
from transformers import GPT2Tokenizer
from genre_conditioned_model import GenreConditionedModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import wandb
from peft import PeftModel

# 🎯 W&B Setup
wandb.init(project="tag_gen", name="inference-genre")

# === Paths ===
MODEL_DIR = "gpt2-output-genre"
TOKENIZER_DIR = "tokenizer"
GENRE_LIST_PATH = "genre_list.txt"
EVAL_FILE = "eval_genre.csv"

# === Load genre list ===
with open(GENRE_LIST_PATH, "r") as f:
    genre_list = [line.strip() for line in f if line.strip()]

# === Load model + tokenizer ===
print("🚀 Loading genre-conditioned model...")
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_DIR)

model = GenreConditionedModel(
    base_model_name="gpt2",  # or your original base model name
    genre_list=genre_list,
)
model.peft_model = PeftModel.from_pretrained(model.peft_model, MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Load eval data ===
print("📂 Loading evaluation set...")
df = pd.read_csv(EVAL_FILE)
df = df[df["overview"].notna() & df["tagline"].notna() & df["genre"].notna()].head(100)

# === Init metrics ===
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
generated_list = []
reference_list = []

# === Run inference ===
print("🔎 Running inference...")
for _, row in df.iterrows():
    title = row["title"]
    overview = row["overview"]
    reference = row["tagline"]
    
    # Support multiple genres (stored as stringified lists)
    genre_raw = row["genre"]
    try:
        genre = ast.literal_eval(genre_raw) if isinstance(genre_raw, str) else genre_raw
    except Exception:
        genre = []

    # Validate against genre_list
    genre = [g for g in genre if g in genre_list]
    if not genre:
        genre = ["unknown"]

    input_text = f"Overview: {overview}\nTagline:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output = model.generate_with_genre(
            **inputs,
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=False,
            length_penalty=1.0,
            genre=[genre]  # Pass genre as batch list of lists
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_tagline = decoded
    if "Tagline:" in decoded:
        generated_tagline = decoded.split("Tagline:")[-1]
    generated_tagline = generated_tagline.strip().split("\n")[0].strip()

    import re

    # Truncate generated tagline at the first punctuation mark (if any)
    match = re.search(r"(.+?[.!?])\s", generated_tagline)
    if match:
        generated_tagline = match.group(1).strip()


    for artifact in ["Overview:", "overview", "OVERVIEW"]:
        if generated_tagline.lower().startswith(artifact.lower()):
            generated_tagline = generated_tagline[len(artifact):].strip()

    generated_list.append(generated_tagline)
    reference_list.append(reference)

# === Evaluation ===
print("📊 Evaluating ROUGE...")
rouge1_scores, rougeL_scores = [], []
for ref, hyp in zip(reference_list, generated_list):
    scores = rouge.score(ref, hyp)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

print("📊 Evaluating BERTScore...")
P, R, F1 = bert_score(generated_list, reference_list, lang="en", verbose=True)

# === Logging ===
print("\n📈 Evaluation Summary:")
print(f"Avg ROUGE-1 F1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"Avg ROUGE-L F1: {sum(rougeL_scores)/len(rougeL_scores):.4f}")
print(f"Avg BERTScore F1: {F1.mean().item():.4f}")

wandb.log({
    "avg_rouge1_f1": sum(rouge1_scores)/len(rouge1_scores),
    "avg_rougeL_f1": sum(rougeL_scores)/len(rougeL_scores),
    "avg_bertscore_f1": F1.mean().item()
})

# === Save Outputs ===
output_df = pd.DataFrame({
    "Title": df["title"].tolist(),
    "Genre": df["genre"].tolist(),
    "Original": reference_list,
    "Generated": generated_list
})
output_df.to_csv("generated_vs_original_genre.csv", index=False)
print("✅ Output saved to generated_vs_original_genre.csv")
