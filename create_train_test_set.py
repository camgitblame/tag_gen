import pandas as pd
from sklearn.model_selection import train_test_split
from ast import literal_eval
import os

# === Safely parse genres from JSON-like strings ===
def parse_genres(genre_str):
    try:
        genres = literal_eval(genre_str)
        if isinstance(genres, list):
            return [g["name"] for g in genres if "name" in g]
    except:
        return []
    return []

# === Format for GPT-2 input: Overview + Tagline combined ===
def format_for_gpt2(df):
    df["prompt"] = "Overview: " + df["overview"].str.strip() + "\nTagline:"
    df["formatted"] = df["prompt"] + " " + df["tagline"].str.strip()
    return df[df["formatted"].str.len() < 1000]  # Optional: filter overly long examples

# === Main processing function ===
def create_balanced_train_test(input_path, train_out, test_out, test_frac=0.2, random_seed=42):
    print("🚀 Script started")
    print("📂 Loading dataset...")
    df = pd.read_csv(input_path, low_memory=False)

    # Clean missing or empty overview/tagline entries
    df = df[df["overview"].notna() & df["tagline"].notna()]
    df = df[df["overview"].str.strip() != ""]
    df = df[df["tagline"].str.strip() != ""]

    # Parse genres
    df["parsed_genres"] = df["genres"].apply(parse_genres)
    df = df[df["parsed_genres"].map(len) > 0]
    df["genre"] = df["parsed_genres"].apply(lambda g: ", ".join(g))  # Convert list to comma-separated string

    # Use first genre only for stratification
    df["primary_genre"] = df["parsed_genres"].apply(lambda g: g[0])

    # Stratified train-test split using primary genre
    train_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        stratify=df["primary_genre"],
        random_state=random_seed
    )

    print(f"✅ Final split — Train: {len(train_df)}, Test: {len(test_df)}")

    # === Save cleaned and formatted CSVs ===
    # === Save cleaned and formatted CSVs ===
    train_final = train_df[["title", "overview", "tagline", "genre"]]
    test_final = test_df[["title", "overview", "tagline", "genre"]]

    train_final.to_csv(train_out, index=False)
    test_final.to_csv(test_out, index=False)

    print(f"📊 Train set saved to {train_out} with {len(train_final)} rows")
    print(f"📊 Test set saved to {test_out} with {len(test_final)} rows")

    # === Generate GPT-2 files ===
    print("📝 Generating GPT-2 training and eval files...")
    formatted_train = format_for_gpt2(train_df)
    formatted_test = test_df[test_df["overview"].notna() & test_df["tagline"].notna()]

    formatted_train["formatted"].to_csv("train.txt", index=False, header=False)
    formatted_test[["title", "overview", "tagline"]].to_csv("eval.csv", index=False)
    formatted_test[["title", "overview", "tagline", "genre"]].to_csv("eval_with_genre.csv", index=False)

    print("✅ Files saved:")
    print(" - train.csv")
    print(" - test.csv")
    print(" - train.txt")
    print(" - eval.csv")
    print(" - eval_with_genre.csv")

# === Run the processing if executed directly ===
if __name__ == "__main__":
    create_balanced_train_test(
        input_path="movies_metadata.csv",
        train_out="train.csv",
        test_out="test.csv",
        test_frac=0.2
    )