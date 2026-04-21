"""
test_validate.py — unit tests for the validation module.
Run: pytest tests/test_validate.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.validate import (
    _check_min_rows,
    _check_null_rows,
    _check_required_cols,
    _check_target,
    _check_class_balance,
    CKD_REQUIRED_COLS,
    CKD_TARGET_VALUES,
    THYROID_TARGET_VALUES,
    BALANCE_MIN_PCT,
)


def make_ckd_df(n=400):
    """Minimal valid CKD dataframe matching the actual CSV structure."""
    np.random.seed(42)
    return pd.DataFrame({
        "age":   np.random.uniform(20, 80, n),
        "bp":    np.random.uniform(60, 140, n),
        "sg":    np.random.uniform(1.005, 1.025, n),
        "al":    np.random.randint(0, 5, n).astype(float),
        "su":    np.random.randint(0, 5, n).astype(float),
        "bgr":   np.random.uniform(70, 200, n),
        "bu":    np.random.uniform(10, 60, n),
        "sc":    np.random.uniform(0.6, 3.0, n),
        "sod":   np.random.uniform(130, 150, n),
        "pot":   np.random.uniform(3.0, 6.0, n),
        "hemo":  np.random.uniform(8, 17, n),
        "pcv":   np.random.uniform(25, 50, n),
        "wc":    np.random.uniform(4000, 12000, n),
        "rc":    np.random.uniform(3.0, 6.0, n),
        "htn":   np.random.randint(0, 2, n).astype(float),
        "dm":    np.random.randint(0, 2, n).astype(float),
        "cad":   np.random.randint(0, 2, n).astype(float),
        "appet": np.random.randint(0, 2, n).astype(float),
        "pe":    np.random.randint(0, 2, n).astype(float),
        "ane":   np.random.randint(0, 2, n).astype(float),
        "class": np.where(np.random.rand(n) > 0.4, "ckd", "notckd"),
    })


#  _check_min_rows 

class TestCheckMinRows:
    def test_passes_when_enough_rows(self):
        df     = make_ckd_df(400)
        errors = []
        _check_min_rows(df, 300, "CKD", errors)
        assert errors == []

    def test_fails_when_too_few_rows(self):
        df     = make_ckd_df(50)
        errors = []
        _check_min_rows(df, 300, "CKD", errors)
        assert len(errors) == 1
        assert "50 rows" in errors[0]

    def test_passes_at_exact_minimum(self):
        df     = make_ckd_df(300)
        errors = []
        _check_min_rows(df, 300, "CKD", errors)
        assert errors == []


# _check_null_rows 

class TestCheckNullRows:
    def test_passes_with_no_null_rows(self):
        df     = make_ckd_df(10)
        errors = []
        _check_null_rows(df, "CKD", errors)
        assert errors == []

    def test_fails_with_fully_null_row(self):
        df          = make_ckd_df(10)
        df.iloc[0]  = np.nan
        errors      = []
        _check_null_rows(df, "CKD", errors)
        assert len(errors) == 1
        assert "fully-null" in errors[0]

    def test_partial_nulls_do_not_trigger(self):
        df             = make_ckd_df(10)
        df.loc[0, "sc"] = np.nan
        errors         = []
        _check_null_rows(df, "CKD", errors)
        assert errors == []


# _check_required_cols

class TestCheckRequiredCols:
    def test_passes_all_cols_present(self):
        df     = make_ckd_df(10)
        errors = []
        _check_required_cols(df, ["age", "bp", "sc", "bu"], [], "CKD", errors)
        assert errors == []

    def test_fails_missing_col(self):
        df     = make_ckd_df(10).drop(columns=["sc"])
        errors = []
        _check_required_cols(df, ["sc", "bu"], [], "CKD", errors)
        assert any("sc" in e for e in errors)

    def test_flexible_passes_when_first_alias_present(self):
        df     = make_ckd_df(10)          # has 'wc'
        errors = []
        _check_required_cols(df, [], [("wbcc", "wc")], "CKD", errors)
        assert errors == []

    def test_flexible_passes_when_second_alias_present(self):
        df         = make_ckd_df(10)
        df         = df.rename(columns={"wc": "wbcc"})
        errors     = []
        _check_required_cols(df, [], [("wbcc", "wc")], "CKD", errors)
        assert errors == []

    def test_flexible_fails_when_neither_alias_present(self):
        df     = make_ckd_df(10).drop(columns=["wc"])
        errors = []
        _check_required_cols(df, [], [("wbcc", "wc")], "CKD", errors)
        assert len(errors) == 1

    def test_multiple_missing_cols_all_reported(self):
        df     = make_ckd_df(10).drop(columns=["sc", "bu", "hemo"])
        errors = []
        _check_required_cols(df, ["sc", "bu", "hemo"], [], "CKD", errors)
        assert len(errors) == 3


#  _check_target

class TestCheckTarget:
    def test_passes_valid_ckd_targets(self):
        df     = make_ckd_df(10)
        errors = []
        _check_target(df, "class", CKD_TARGET_VALUES, "CKD", errors)
        assert errors == []

    def test_fails_when_target_col_missing(self):
        df     = make_ckd_df(10).drop(columns=["class"])
        errors = []
        _check_target(df, "class", CKD_TARGET_VALUES, "CKD", errors)
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_warns_but_does_not_fail_on_unexpected_labels(self):
        # Unexpected labels are warned but not raised as errors
        # (they get dropped in preprocessing)
        df               = make_ckd_df(10)
        df.loc[0, "class"] = "unknown_label"
        errors           = []
        _check_target(df, "class", CKD_TARGET_VALUES, "CKD", errors)
        assert errors == []

    def test_passes_valid_thyroid_targets(self):
        df = pd.DataFrame({
            "target": ["negative", "primary hypothyroid",
                       "compensated hypothyroid", "T3 toxic"]
        })
        errors = []
        _check_target(df, "target", THYROID_TARGET_VALUES, "Thyroid", errors)
        assert errors == []


# _check_class_balance 

class TestCheckClassBalance:
    def test_passes_balanced_dataset(self):
        df     = make_ckd_df(400)
        errors = []
        _check_class_balance(df, "class", "CKD", errors)
        assert errors == []

    def test_fails_severely_imbalanced(self):
        df           = make_ckd_df(400)
        df["class"]  = "ckd"
        df.loc[:3, "class"] = "notckd"   # only 1% minority
        errors       = []
        _check_class_balance(df, "class", "CKD", errors)
        assert len(errors) == 1
        assert "minority" in errors[0]

    def test_passes_at_exact_threshold(self):
        # Create exactly BALANCE_MIN_PCT minority
        n        = 100
        n_min    = int(n * BALANCE_MIN_PCT / 100)
        df       = pd.DataFrame({
            "class": ["notckd"] * (n - n_min) + ["ckd"] * n_min
        })
        errors   = []
        _check_class_balance(df, "class", "CKD", errors)
        assert errors == []

    def test_handles_missing_target_col_gracefully(self):
        df     = make_ckd_df(10)
        errors = []
        _check_class_balance(df, "nonexistent", "CKD", errors)
        assert errors == []