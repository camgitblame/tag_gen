import pandas as pd

# 📂 Load the original dataset
print("📂 Loading original dataset...")
df = pd.read_csv("movies_metadata.csv", low_memory=False)

# 🧼 Clean and filter rows
df = df[df["overview"].notna() & df["tagline"].notna()]
df = df[df["overview"].str.strip() != ""]
df = df[df["tagline"].str.strip() != ""]
df = df.drop_duplicates(subset=["overview", "tagline"])

# 🧪 Sample 100 rows for evaluation
eval_df = df[["title", "overview", "tagline"]].head(100)

# 💾 Save as eval.csv
eval_df.to_csv("eval.csv", index=False)
print(f"✅ Saved {len(eval_df)} rows to eval.csv")
