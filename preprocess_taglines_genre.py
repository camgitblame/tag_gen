import pandas as pd
import ast
from transformers import GPT2Tokenizer


def load_and_clean(path="movies_metadata.csv"):
    print("[📂] Loading dataset...")
    df = pd.read_csv(path, low_memory=False)

    df = df[df["overview"].notna() & df["tagline"].notna()]
    df = df[df["overview"].str.strip() != ""]
    df = df[df["tagline"].str.strip() != ""]
    df = df.drop_duplicates(subset=["overview", "tagline"])
    return df


# Collect all genres from the column
def extract_and_collect_genres(df):
    genre_set = set()
    primary_genres = []

    for genre_str in df["genres"]:
        try:
            genres = ast.literal_eval(genre_str)
            if isinstance(genres, list) and genres:
                names = [g["name"].lower() for g in genres if "name" in g]
                genre_set.update(names)
                primary_genres.append(names)
            else:
                primary_genres.append(None)
        except Exception:
            primary_genres.append(None)

    df["genre"] = primary_genres
    df = df[df["genre"].notna()]
    return df, sorted(list(genre_set))


def format_with_genre(df):
    print("[🪄] Formatting rows and extracting genre...")
    df["overview"] = df["overview"].str.strip()
    df["tagline"] = df["tagline"].str.strip()
    df = df[df["overview"].str.len() + df["tagline"].str.len() < 1000]  # Limit combined length
    return df[["overview", "tagline", "genre"]]


def save_json(df, output_path="train_with_genre.json"):
    print(f"[💾] Saving {len(df)} examples to {output_path}...")
    df.to_json(output_path, orient="records", lines=True)


def save_eval_csv(df, output_path="eval_genre.csv", num_samples=100):
    print(f"[📁] Saving {num_samples} examples to {output_path}...")
    eval_df = df[["title", "genre", "overview", "tagline"]].head(num_samples).copy()
    eval_df.to_csv(output_path, index=False)


def setup_tokenizer(tokenizer_path="gpt2", save_path="tokenizer/"):
    print("[🧠] Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(save_path)
    print(f"[💾] Tokenizer saved to {save_path}")
    return tokenizer


if __name__ == "__main__":
    df = load_and_clean("movies_metadata.csv")
    df, genre_list = extract_and_collect_genres(df)
    df_formatted = format_with_genre(df)

    save_json(df_formatted, "train_with_genre.json")
    save_eval_csv(df, output_path="eval_genre.csv", num_samples=100)

    with open("genre_list.txt", "w") as f:
        for genre in genre_list:
            f.write(f"{genre}\n")

    setup_tokenizer("gpt2", "tokenizer/")
    print("[✅] Preprocessing complete.")
    print(f"[📚] Extracted {len(genre_list)} genres: {genre_list}")
