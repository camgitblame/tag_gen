import pandas as pd

df = pd.read_csv("eval.csv")

# Check for <sep> tokens or malformed rows
has_sep = df["overview"].str.contains("<sep>").any()
missing_cols = not {"overview", "tagline"}.issubset(df.columns)

if missing_cols:
    print("❌ Missing required columns: 'overview' and/or 'tagline'")
elif has_sep:
    print("❌ Found '<sep>' tokens — incompatible with new format.")
else:
    print("✅ eval.csv is compatible with the new inference format.")
