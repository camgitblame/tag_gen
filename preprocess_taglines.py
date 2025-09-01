import pandas as pd
from transformers import GPT2Tokenizer


# Load and clean the raw dataset
def load_and_clean(path="movies_metadata.csv"):
    print("Loading dataset...")
    df = pd.read_csv(path, low_memory=False)

    df = df[df["overview"].notna() & df["tagline"].notna()]
    df = df[df["overview"].str.strip() != ""]
    df = df[df["tagline"].str.strip() != ""]
    df = df.drop_duplicates(subset=["overview", "tagline"])
    return df


# Format as "Overview: ... \nTagline: ..."
def format_with_prompt(df):
    print("Formatting rows for tagline generation...")
    df["prompt"] = "Overview: " + df["overview"].str.strip() + "\nTagline:"
    df["formatted"] = df["prompt"] + " " + df["tagline"].str.strip()
    df = df[df["formatted"].str.len() < 1000]
    return df


# Save to train.txt
def save_txt(df, output_path="train.txt"):
    print(f"Saving {len(df)} examples to {output_path}...")
    df["formatted"].to_csv(output_path, index=False, header=False)


# Load tokenizer
def setup_tokenizer(tokenizer_path="gpt2", save_path="tokenizer/"):
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer


# Run the full pipeline
if __name__ == "__main__":
    df = load_and_clean("movies_metadata.csv")
    df = format_with_prompt(df)
    save_txt(df, "train.txt")
    setup_tokenizer("gpt2", "tokenizer/")
    print("[Complete] Preprocessing complete.")
