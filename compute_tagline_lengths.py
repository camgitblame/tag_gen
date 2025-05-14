import pandas as pd
from transformers import GPT2Tokenizer

# === Load dataset ===
df = pd.read_csv("movies_metadata.csv")
df = df[df["tagline"].notna() & df["tagline"].str.strip().ne("")]

# === Load tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# === Compute token lengths ===
df["token_length"] = df["tagline"].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))

# === Calculate summary stats ===
stats = {
    "average_token_length": df["token_length"].mean(),
    "max_token_length": df["token_length"].max(),
    "min_token_length": df["token_length"].min(),
    "std_token_length": df["token_length"].std(),
    "total_samples": len(df)
}

# === Save summary to CSV ===
summary_df = pd.DataFrame([stats])
summary_df.columns = [
    "Avg Token Length",
    "Max Token Length",
    "Min Token Length",
    "Std Token Length",
    "Total Samples"
]
summary_df.to_csv("tagline_token_stats.csv", index=False)

print("✅ Stats saved to 'tagline_token_stats.csv' ")
