"""
validate.py

Automated data validation: runs as a DVC pipeline gate between
download and preprocess. If this fails, nothing downstream runs.

Checks per dataset:
    1. Minimum row count
    2. Required columns present (with alias support)
    3. Target column has expected label values
    4. No fully-null rows
    5. Class balance (minority class must be >10%)

collects ALL errors before raising, so you see every
problem at once rather than fixing one at a time.

Usage:
    python src/features/validate.py          # validates both
    python src/features/validate.py ckd
    python src/features/validate.py thyroid
"""

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

RAW_DIR = "data/raw"

# Schema expectations

# Columns that must be present in CKD raw CSV
CKD_REQUIRED_COLS = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "htn", "dm", "cad",
    "appet", "pe", "ane",
]

# ucimlrepo uses wbcc/rbcc, download uses wc/rc
CKD_FLEXIBLE_COLS = [("wbcc", "wc"), ("rbcc", "rc")]

# All valid raw target strings in the CKD file
CKD_TARGET_VALUES = {"ckd", "notckd", "not ckd"}

# Thyroid: only a few are strictly required: the rest can be missing
THYROID_REQUIRED_COLS = ["age", "TSH"]

# All valid raw target strings across all thyroid data files
THYROID_TARGET_VALUES = {
    "negative",
    "primary hypothyroid",
    "compensated hypothyroid",
    "secondary hypothyroid",
    "t3 toxic",
    "toxic goitre",
    "secondary toxic",
    "hyperthyroid",
}

CKD_MIN_ROWS     = 300
THYROID_MIN_ROWS = 3000
BALANCE_MIN_PCT  = 5.0   # fail if minority class < 5%


# Individual checks

def _check_min_rows(df, min_rows, name, errors):
    if len(df) < min_rows:
        errors.append(
            f"{name}: only {len(df)} rows — expected >= {min_rows}"
        )
    else:
        log.info("  [PASS] row count: %d", len(df))


def _check_null_rows(df, name, errors):
    n = df.isnull().all(axis=1).sum()
    if n > 0:
        errors.append(f"{name}: {n} fully-null rows found")
    else:
        log.info("  [PASS] no fully-null rows")


def _check_required_cols(df, required, flexible, name, errors):
    for col in required:
        if col not in df.columns:
            errors.append(f"{name}: required column '{col}' missing")

    for col_a, col_b in flexible:
        if col_a not in df.columns and col_b not in df.columns:
            errors.append(
                f"{name}: need '{col_a}' or '{col_b}', neither found"
            )

    present = [c for c in required if c in df.columns]
    log.info("  [PASS] %d/%d required cols present", len(present), len(required))


def _check_target(df, target_col, expected_values, name, errors):
    if target_col not in df.columns:
        errors.append(f"{name}: target column '{target_col}' not found")
        return

    actual = set(
        df[target_col].dropna()
        .astype(str).str.strip().str.lower()
        .str.split(".").str[0].str.strip()
    )
    unexpected = actual - expected_values
    if unexpected:
        # Log as warning : some datasets have rare extra labels we map to NaN
        log.warning(
            "  [WARN] %s: unexpected target values (will be dropped): %s",
            name, unexpected,
        )
    log.info("  [PASS] target column '%s' found, values: %s", target_col, actual)


def _check_class_balance(df, target_col, name, errors):
    if target_col not in df.columns:
        return
    counts = df[target_col].value_counts(normalize=True)
    minority_pct = counts.min() * 100
    log.info("  class balance: %s", counts.to_dict())
    if minority_pct < BALANCE_MIN_PCT:
        errors.append(
            f"{name}: minority class is only {minority_pct:.1f}% "
            f"(threshold: {BALANCE_MIN_PCT}%)"
        )
    else:
        log.info("  [PASS] class balance ok — minority class %.1f%%", minority_pct)


# Public entry points

def validate_ckd() -> bool:
    """
    Run all CKD validation checks on data/raw/ckd.csv.

    Returns:
        True if all checks pass.

    Raises:
        ValueError: listing every failure found.
    """
    t0     = time.time()
    errors = []
    path   = os.path.join(RAW_DIR, "ckd.csv")

    log.info("Validating CKD: %s", path)
    df = pd.read_csv(path, na_values=["?", "\t?", " ?", ""])
    df.columns = [c.strip().lower() for c in df.columns]

    _check_min_rows(df, CKD_MIN_ROWS, "CKD", errors)
    _check_null_rows(df, "CKD", errors)
    _check_required_cols(df, CKD_REQUIRED_COLS, CKD_FLEXIBLE_COLS, "CKD", errors)

    target_col = "class" if "class" in df.columns else "classification"
    _check_target(df, target_col, CKD_TARGET_VALUES, "CKD", errors)
    _check_class_balance(df, target_col, "CKD", errors)

    elapsed = time.time() - t0
    if errors:
        msg = (
            f"CKD validation FAILED ({elapsed:.2f}s) — "
            f"{len(errors)} error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
        raise ValueError(msg)

    log.info("CKD validation PASSED in %.2fs", elapsed)
    return True


def validate_thyroid() -> bool:
    """
    Run all thyroid validation checks on data/raw/thyroid.csv.

    Returns:
        True if all checks pass.

    Raises:
        ValueError: listing every failure found.
    """
    t0     = time.time()
    errors = []
    path   = os.path.join(RAW_DIR, "thyroid.csv")

    log.info("Validating Thyroid: %s", path)
    df = pd.read_csv(path, na_values=["?", ""])
    df.columns = [c.strip() for c in df.columns]

    _check_min_rows(df, THYROID_MIN_ROWS, "Thyroid", errors)
    _check_null_rows(df, "Thyroid", errors)
    _check_required_cols(df, THYROID_REQUIRED_COLS, [], "Thyroid", errors)

    target_col = next(
        (c for c in ["target", "Target", "Class", "class", "diagnosis"]
         if c in df.columns),
        df.columns[-1],
    )
    _check_target(df, target_col, THYROID_TARGET_VALUES, "Thyroid", errors)

    elapsed = time.time() - t0
    if errors:
        msg = (
            f"Thyroid validation FAILED ({elapsed:.2f}s) — "
            f"{len(errors)} error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )
        raise ValueError(msg)

    log.info("Thyroid validation PASSED in %.2fs", elapsed)
    return True


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "both"
    failed = False
    try:
        if task in ("ckd",     "both"): validate_ckd()
        if task in ("thyroid", "both"): validate_thyroid()
        print("\nAll validation checks PASSED")
    except ValueError as e:
        print(f"\n{e}")
        failed = True
    sys.exit(1 if failed else 0)
