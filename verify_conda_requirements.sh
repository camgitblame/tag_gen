#!/bin/bash

echo "🧪 Verifying installed Python packages..."

# === Path to your requirements file ===
REQ_FILE="requirements.txt"

# === Check if requirements.txt exists ===
if [ ! -f "$REQ_FILE" ]; then
    echo "❌ $REQ_FILE not found in current directory: $(pwd)"
    exit 1
fi

# === Activate current virtualenv (optional if already active) ===
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "⚠️ No conda environment is active. Please activate it first."
    exit 1
fi

# === Loop through each requirement and check it ===
MISSING_PKGS=0
while IFS= read -r line; do
    # skip comments and blank lines
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    # extract package name without version specifier
    PKG_NAME=$(echo "$line" | cut -d= -f1)

    pip show "$PKG_NAME" > /dev/null 2>&1
    if [[ $? -ne 0 ]]; then
        echo "❌ Missing: $line"
        ((MISSING_PKGS++))
    else
        echo "✅ Installed: $line"
    fi
done < "$REQ_FILE"

echo "🔎 Package check complete. $MISSING_PKGS missing."

# Exit with 0 if all packages are installed, 1 otherwise
exit $MISSING_PKGS
