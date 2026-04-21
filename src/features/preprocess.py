"""
preprocess.py

Cleans and encodes raw CKD and thyroid CSVs.
Also computes baseline statistics for drift detection.

What this does and why:
    CKD:
        - Maps categorical yes/no/normal/abnormal columns to 1/0
        - Maps target ckd→1, notckd→0
        - Coerces everything to numeric (pd.to_numeric errors='coerce'
          turns unparseable strings into NaN for the imputer to handle)

    Thyroid:
        - Maps boolean flag columns t/f → 1/0
        - Maps sex M/F → 1/0
        - Collapses fine-grained labels to 3 classes (0/1/2)
        - Drops TBG (>95% missing) and referral_source (metadata)

    Baselines:
        - Computes mean, std, min, max, p25, p75 per numeric feature
        - Saved to data/baseline_stats.json
        - Used by FastAPI backend for Prometheus drift gauges

Usage:
    python src/features/preprocess.py ckd
    python src/features/preprocess.py thyroid
    python src/features/preprocess.py baselines
    python src/features/preprocess.py both        # ckd + thyroid + baselines
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)


# CKD

def preprocess_ckd() -> pd.DataFrame:
    """
    Clean and encode the raw CKD dataset.

    Column renames:
        wbcc → wc   (WBC count)
        rbcc → rc   (RBC count)
        class → classification

    Encoding:
        yes/no columns          → 1/0
        normal/abnormal         → 1/0
        present/notpresent      → 1/0
        good/poor               → 1/0
        ckd/notckd target       → 1/0

    Returns:
        Cleaned DataFrame. Also saved to data/processed/ckd_clean.csv
    """
    t0   = time.time()
    path = os.path.join(RAW_DIR, "ckd.csv")
    df   = pd.read_csv(path, na_values=["?", "\t?", " ?", ""])
    df.columns = [c.strip().lower() for c in df.columns]

    log.info("CKD raw: %d rows × %d cols", len(df), len(df.columns))

    # Normalise column names (Different sources use wbcc vs wc, rbcc vs rc, experimented and normalised here)
    df = df.rename(columns={
        "wbcc": "wc",
        "rbcc": "rc",
        "class": "classification",
    })

    # Encode target
    df["classification"] = (
        df["classification"]
        .astype(str).str.strip().str.lower()
        .str.replace("\t", "", regex=False)
        .map({"ckd": 1, "notckd": 0, "not ckd": 0})
    )

    # Encode all remaining object columns
    yn_map = {
        "yes": 1,      "no": 0,
        "normal": 1,   "abnormal": 0,
        "present": 1,  "notpresent": 0,
        "good": 1,     "poor": 0,
        "nan": np.nan,
    }
    for col in df.select_dtypes(include="object").columns:
        if col == "classification":
            continue
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
            .map(yn_map)
        )

    # Coerce all remaining columns to numeric
    for col in df.columns:
        if col != "classification":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["classification"])
    df["classification"] = df["classification"].astype(int)

    # Log missing value summary
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        log.info("CKD missing value summary (imputer handles these):")
        for col, cnt in missing.items():
            log.info("  %-8s %d (%.0f%%)", col, cnt, cnt / len(df) * 100)

    elapsed = time.time() - t0
    rows_per_sec = len(df) / elapsed
    log.info(
        "CKD preprocess done | rows=%d elapsed=%.2fs throughput=%.0f rows/s",
        len(df), elapsed, rows_per_sec,
    )
    log.info("CKD class balance: %s", dict(df["classification"].value_counts()))

    out = os.path.join(PROC_DIR, "ckd_clean.csv")
    df.to_csv(out, index=False)
    log.info("Saved → %s", out)
    return df


# Thyroid

def preprocess_thyroid() -> pd.DataFrame:
    """
    Clean and encode the raw thyroid dataset.

    Label collapse:
        'negative'                  → 0  (Normal)
        contains 'hypo'             → 1  (Hypothyroid)
        contains 'hyper' or 'toxic' → 2  (Hyperthyroid)
        anything else               → dropped

    Encoding:
        Boolean flags (t/f)   → 1/0
        Sex (M/F)             → 1/0

    Dropped columns:
        TBG         — >95% missing across all thyroid files
        TBG_measured
        referral_source — administrative metadata, not a clinical feature

    Returns:
        Cleaned DataFrame. Also saved to data/processed/thyroid_clean.csv
    """
    t0   = time.time()
    path = os.path.join(RAW_DIR, "thyroid.csv")
    df   = pd.read_csv(path, na_values=["?", ""])
    df.columns = [c.strip() for c in df.columns]

    log.info("Thyroid raw: %d rows × %d cols", len(df), len(df.columns))

    # target column
    target_col = df.columns[-1]
    log.info("Target column: '%s'", target_col)

    # Collapse labels
    def _map_label(raw):
        s = str(raw).lower().strip()
        s = s.split(".")[0].split("|")[0].strip()
        if s in ("negative", "normal", "-", ""):
            return 0
        if "hypo" in s:
            return 1
        if "hyper" in s or "toxic" in s:
            return 2
        return np.nan

    df["label"] = df[target_col].apply(_map_label)
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    dropped = before - len(df)
    if dropped > 0:
        log.info("Dropped %d rows with unmapped labels", dropped)

    # Boolean flags: t/f → 1/0
    flag_cols = [
        "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
        "sick", "pregnant", "thyroid_surgery", "I131_treatment",
        "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre",
        "tumor", "hypopituitary", "psych",
        "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured",
        "FTI_measured", "TBG_measured",
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.lower()
                .map({"t": 1, "f": 0, "true": 1, "false": 0,
                      "yes": 1, "no": 0, "nan": np.nan})
            )

    # Sex: M/F → 1/0
    if "sex" in df.columns:
        df["sex"] = (
            df["sex"].astype(str).str.strip().str.upper()
            .map({"M": 1, "F": 0})
        )

    # Numeric coercion
    for col in ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop low-value columns
    drop_cols = [target_col, "referral_source", "TBG", "TBG_measured"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Missing value summary
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        log.info("Thyroid missing value summary:")
        for col, cnt in missing.items():
            log.info("  %-20s %d (%.0f%%)", col, cnt, cnt / len(df) * 100)

    elapsed = time.time() - t0
    rows_per_sec = len(df) / elapsed
    log.info(
        "Thyroid preprocess done | rows=%d elapsed=%.2fs throughput=%.0f rows/s",
        len(df), elapsed, rows_per_sec,
    )
    log.info(
        "Label dist: Normal=%d  Hypo=%d  Hyper=%d",
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
        (df["label"] == 2).sum(),
    )

    out = os.path.join(PROC_DIR, "thyroid_clean.csv")
    df.to_csv(out, index=False)
    log.info("Saved → %s", out)
    return df


# Baselines

def compute_baselines():
    """
    Compute per-feature baseline statistics from both cleaned datasets.

    Statistics: mean, std, min, max, p25, p75, count.
    Saved to data/baseline_stats.json.

    The FastAPI backend loads this file at startup and uses these values
    to populate Prometheus drift gauges — comparing live request values
    against training distribution.
    """
    stats = {}
    for name, path in [
        ("ckd",     os.path.join(PROC_DIR, "ckd_clean.csv")),
        ("thyroid", os.path.join(PROC_DIR, "thyroid_clean.csv")),
    ]:
        if not os.path.exists(path):
            log.warning("Skipping baselines for %s — file not found", name)
            continue
        df = pd.read_csv(path)
        stats[name] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            stats[name][col] = {
                "mean":  round(float(series.mean()),            4),
                "std":   round(float(series.std()),             4),
                "min":   round(float(series.min()),             4),
                "max":   round(float(series.max()),             4),
                "p25":   round(float(series.quantile(0.25)),    4),
                "p75":   round(float(series.quantile(0.75)),    4),
                "count": int(len(series)),
            }
        log.info("Baselines: %s — %d features", name, len(stats[name]))

    out = "data/baseline_stats.json"
    with open(out, "w") as fh:
        json.dump(stats, fh, indent=2)
    log.info("Baseline stats saved → %s", out)


# Entry point
if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "both"
    if task in ("ckd",      "both"): preprocess_ckd()
    if task in ("thyroid",  "both"): preprocess_thyroid()
    if task in ("baselines","both"): compute_baselines()