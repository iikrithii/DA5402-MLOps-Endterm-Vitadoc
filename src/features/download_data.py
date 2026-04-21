"""
download_data.py

Reliable data download using the official UCI ML Repo Python client.
Usage: python src/features/download_data.py
"""
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
import urllib.request

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

#  CKD 
print("Downloading CKD dataset from UCI")
ckd_dataset = fetch_ucirepo(id=336)
ckd_df = pd.concat([ckd_dataset.data.features, ckd_dataset.data.targets], axis=1)
ckd_df['class'] = (
    ckd_df['class']
    .astype(str)
    .str.strip()        # removes \t, spaces, etc.
    .str.lower()
)
ckd_df.to_csv(os.path.join(RAW_DIR, "ckd.csv"), index=False)
print(f"CKD: {len(ckd_df)} rows, columns: {list(ckd_df.columns)}")

# Thyroid: download two files and combine to get all 3 classes
# allhypo.data  → normal + hypothyroid cases
# allhyper.data → normal + hyperthyroid cases
# Together they give Normal / Hypo / Hyper with enough samples

THYROID_COLS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
    "referral_source", "target",
]

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease"

dfs = []
for filename in ["allhypo.data", "allhyper.data", "allhypo.test", "allhyper.test"]:
    url      = f"{BASE_URL}/{filename}"
    out_path = os.path.join(RAW_DIR, filename)
    print(f"  Downloading {filename}")
    try:
        urllib.request.urlretrieve(url, out_path)
        df = pd.read_csv(out_path, header=None, names=THYROID_COLS, na_values="?")
        # Strip trailing metadata from target: "negative.  |  1" → "negative"
        df["target"] = df["target"].astype(str).str.split(r"\.").str[0].str.strip()
        dfs.append(df)
        print(f"    {len(df)} rows, targets: {df['target'].value_counts().to_dict()}")
    except Exception as e:
        print(f"    Warning: could not download {filename}: {e}")

combined = pd.concat(dfs, ignore_index=True)

# Drop exact duplicate rows (same patient appears in both .data and .test)
combined = combined.drop_duplicates()
combined.to_csv(os.path.join(RAW_DIR, "thyroid.csv"), index=False)
print(f"\nThyroid combined: {len(combined)} rows")
print(f"Target distribution:\n{combined['target'].value_counts()}")

print("Files in data/raw/:")
for f in os.listdir(RAW_DIR):
    size = os.path.getsize(os.path.join(RAW_DIR, f)) // 1024
    print(f"{f} ({size} KB)")