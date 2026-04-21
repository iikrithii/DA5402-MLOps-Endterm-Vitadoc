"""
engineer.py

Feature engineering — runs after preprocess, before training.
Kept separate from model logic so feature versioning is independent.

New features:

    CKD → ckd_engineered.csv:
        kidney_stress  float [0-1]
            Weighted composite of normalised creatinine (40%), blood urea (35%),
            and inverted haemoglobin (25%).
            Rationale: CKD causes sc↑ + bu↑ + hemo↓ simultaneously. Each value
            alone can sit within normal range in early-stage CKD. The composite
            catches the joint pattern that no single threshold captures.

    Thyroid → thyroid_engineered.csv:
        tsh_t3_ratio   float
            TSH / (T3 + 0.01) to avoid division by zero.
            Rationale: primary hypothyroidism → TSH↑, T3↓ → ratio very high.
            Secondary hypothyroidism → TSH↓ (pituitary failure), T3↓ → ratio ~1.
            A plain TSH threshold misclassifies every secondary case.
            The ratio distinguishes them. 

Inspired from github repositroies that do feature engineering 
on the same datasets taken for this task. 

Usage:
    python src/features/engineer.py          # both datasets
    python src/features/engineer.py ckd
    python src/features/engineer.py thyroid
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

PROC_DIR = "data/processed"


def engineer_ckd(
    input_path:  str = None,
    output_path: str = None,
    save:        bool = True,
) -> pd.DataFrame:
    """
    Add kidney_stress engineered feature to the CKD dataset.

    Scaling uses population-level clinical ranges as bounds so the
    score is interpretable across datasets, not just this one sample.

    Args:
        input_path:  reads from ckd_clean.csv if None.
        output_path: writes to ckd_engineered.csv if None.

    Returns:
        DataFrame with kidney_stress column appended.
    """
    output_path = output_path or os.path.join(PROC_DIR, "ckd_engineered.csv")
    df = input_path if isinstance(input_path, pd.DataFrame) else pd.read_csv(input_path or os.path.join(PROC_DIR, "ckd_clean.csv"))


    log.info("CKD engineer | %d rows × %d cols", len(df), len(df.columns))

    components = []

    if "sc" in df.columns:
        # Normal upper bound ~1.3, severe CKD can reach ~20
        sc_norm = (df["sc"] - 0.7) / (20.0 - 0.7)
        components.append(sc_norm.clip(0, 1) * 0.40)

    if "bu" in df.columns:
        # Normal upper bound ~25, uraemia can reach ~300
        bu_norm = (df["bu"] - 7.0) / (300.0 - 7.0)
        components.append(bu_norm.clip(0, 1) * 0.35)

    if "hemo" in df.columns:
        # Invert: low haemoglobin = high kidney stress
        # Range 2–17 g/dL covers severe anaemia to normal
        hemo_inv = (17.0 - df["hemo"]) / (17.0 - 2.0)
        components.append(hemo_inv.clip(0, 1) * 0.25)

    if not components:
        log.warning("CKD: sc/bu/hemo all missing — kidney_stress not added")
        if save:
            df.to_csv(output_path, index=False)
        return df

    df["kidney_stress"] = sum(components)

    log.info("CKD engineer done | %d rows × %d cols", len(df), len(df.columns))
    if save:
        df.to_csv(output_path, index=False)
        log.info("Saved → %s", output_path)
    return df


def engineer_thyroid(
    input_path:  str = None,
    output_path: str = None,
    save:        bool = True,
) -> pd.DataFrame:
    """
    Add tsh_t3_ratio engineered feature to the thyroid dataset.

    tsh_t3_ratio = TSH / (T3 + 0.01)

    The epsilon (0.01) prevents division by zero when T3 is missing
    or near zero, which happens in some hyperthyroid cases.

    Args:
        input_path:  reads from thyroid_clean.csv if None.
        output_path: writes to thyroid_engineered.csv if None.

    Returns:
        DataFrame with tsh_t3_ratio column appended.
    """
    output_path = output_path or os.path.join(PROC_DIR, "thyroid_engineered.csv")
    df = input_path if isinstance(input_path, pd.DataFrame) else pd.read_csv(input_path or os.path.join(PROC_DIR, "thyroid_clean.csv"))


    log.info("Thyroid engineer | %d rows × %d cols", len(df), len(df.columns))

    if "TSH" not in df.columns or "T3" not in df.columns:
        log.warning("Thyroid: TSH or T3 missing — tsh_t3_ratio not added")
        if save:
            df.to_csv(output_path, index=False)
        return df

    epsilon= 0.01
    df["tsh_t3_ratio"] = df["TSH"] / (df["T3"] + epsilon)

    log.info("Thyroid engineer done | %d rows × %d cols", len(df), len(df.columns))
    if save:
        df.to_csv(output_path, index=False)
        log.info("Saved → %s", output_path)
    return df


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "both"
    if task in ("ckd",     "both"): engineer_ckd()
    if task in ("thyroid", "both"): engineer_thyroid()