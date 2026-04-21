"""
test_engineer.py — unit tests for the feature engineering module.
Run: pytest tests/test_engineer.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.engineer import engineer_ckd, engineer_thyroid


def make_ckd_df():
    return pd.DataFrame({
        "age":            [48.0, 25.0, 60.0, 35.0],
        "sc":             [3.5,  0.9,  0.5,  2.0],
        "hemo":           [7.0,  14.5, 17.0, 9.0],
        "bu":             [90.0, 15.0, 6.0,  55.0],
        "classification": [1,    0,    1,    1],
    })


def make_thyroid_df():
    return pd.DataFrame({
        "age":   [41.0, 23.0, 55.0, 30.0],
        "TSH":   [0.2,  2.5,  18.0, 1.0],
        "T3":    [2.8,  1.5,  0.4,  1.2],
        "label": [2,    0,    1,    0],
    })


#  engineer_ckd 

class TestEngineerCKD:
    def test_kidney_stress_column_added(self):
        result = engineer_ckd(make_ckd_df(), save=False)
        assert "kidney_stress" in result.columns

    def test_kidney_stress_within_zero_one(self):
        result = engineer_ckd(make_ckd_df(), save=False)
        vals   = result["kidney_stress"].dropna()
        assert (vals >= 0).all() and (vals <= 1.01).all(), \
            f"Out of range: {vals.values}"

    def test_kidney_stress_higher_for_ckd_on_average(self):
        """CKD patients have higher sc, higher bu, lower hemo → higher stress."""
        np.random.seed(0)
        n       = 200
        ckd     = pd.DataFrame({
            "sc":             np.random.uniform(2.0, 8.0, n),
            "bu":             np.random.uniform(50,  200, n),
            "hemo":           np.random.uniform(6,   11,  n),
            "classification": [1] * n,
        })
        healthy = pd.DataFrame({
            "sc":             np.random.uniform(0.6, 1.2, n),
            "bu":             np.random.uniform(8,   20,  n),
            "hemo":           np.random.uniform(13,  17,  n),
            "classification": [0] * n,
        })
        df      = pd.concat([ckd, healthy], ignore_index=True)
        result  = engineer_ckd(df, save=False)
        ckd_mean    = result[result["classification"] == 1]["kidney_stress"].mean()
        notckd_mean = result[result["classification"] == 0]["kidney_stress"].mean()
        assert ckd_mean > notckd_mean, \
            f"CKD mean={ckd_mean:.3f} should > NotCKD mean={notckd_mean:.3f}"

    def test_kidney_stress_not_perfectly_correlated_with_sc(self):
        """
        kidney_stress combines three features — should not be
        a near-perfect linear copy of any single one.
        """
        np.random.seed(1)
        n  = 300
        df = pd.DataFrame({
            "sc":             np.random.uniform(0.5, 10.0, n),
            "bu":             np.random.uniform(5,   200,  n),
            "hemo":           np.random.uniform(5,   17,   n),
            "classification": np.random.randint(0, 2, n),
        })
        result  = engineer_ckd(df, save=False)
        corr_sc = result["kidney_stress"].corr(result["sc"])
        assert abs(corr_sc) < 0.95, \
            f"kidney_stress too correlated with sc: r={corr_sc:.3f}"

    def test_original_columns_preserved(self):
        df     = make_ckd_df()
        result = engineer_ckd(df.copy(), save=False)
        for col in df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self):
        df     = make_ckd_df()
        result = engineer_ckd(df.copy(), save=False)
        assert len(result) == len(df)

    def test_handles_missing_sc_gracefully(self):
        """If sc is missing, kidney_stress still computed from bu + hemo."""
        df     = make_ckd_df().drop(columns=["sc"])
        result = engineer_ckd(df, save=False)
        assert "kidney_stress" in result.columns
        assert result["kidney_stress"].isna().sum() == 0

    def test_handles_all_components_missing(self):
        """If sc, bu, hemo all absent, kidney_stress is not added."""
        df     = make_ckd_df().drop(columns=["sc", "bu", "hemo"])
        result = engineer_ckd(df, save=False)
        assert "kidney_stress" not in result.columns

    def test_no_new_nulls_introduced(self):
        """Engineering should not introduce NaNs where there were none."""
        df     = make_ckd_df()   # no nulls in fixture
        result = engineer_ckd(df.copy(), save=False)
        assert result["kidney_stress"].isna().sum() == 0

    def test_accepts_dataframe_directly(self):
        """Passing a DataFrame as first arg should not attempt file read."""
        df = make_ckd_df()
        # If this raises FileNotFoundError the isinstance guard is missing
        result = engineer_ckd(df, save=False)
        assert isinstance(result, pd.DataFrame)


# engineer_thyroid 

class TestEngineerThyroid:
    def test_tsh_t3_ratio_added(self):
        result = engineer_thyroid(make_thyroid_df(), save=False)
        assert "tsh_t3_ratio" in result.columns

    def test_no_division_by_zero_when_t3_is_zero(self):
        df       = make_thyroid_df()
        df["T3"] = 0.0
        result   = engineer_thyroid(df, save=False)
        assert np.isfinite(result["tsh_t3_ratio"]).all()
        assert result["tsh_t3_ratio"].isna().sum() == 0

    def test_ratio_higher_for_hypothyroid_than_normal(self):
        """
        Primary hypothyroid: TSH high, T3 low → ratio very high.
        Normal: TSH normal, T3 normal → ratio moderate.
        """
        np.random.seed(42)
        n      = 200
        normal = pd.DataFrame({
            "TSH":   np.random.uniform(0.5, 3.5, n),
            "T3":    np.random.uniform(1.0, 2.0, n),
            "label": [0] * n,
        })
        hypo   = pd.DataFrame({
            "TSH":   np.random.uniform(8.0, 50.0, n),
            "T3":    np.random.uniform(0.2, 0.8,  n),
            "label": [1] * n,
        })
        df     = pd.concat([normal, hypo], ignore_index=True)
        result = engineer_thyroid(df, save=False)
        normal_mean = result[result["label"] == 0]["tsh_t3_ratio"].mean()
        hypo_mean   = result[result["label"] == 1]["tsh_t3_ratio"].mean()
        assert hypo_mean > normal_mean, \
            f"Hypo mean={hypo_mean:.2f} should > Normal mean={normal_mean:.2f}"

    def test_ratio_not_perfectly_correlated_with_tsh(self):
        """
        The ratio encodes TSH×T3 interaction.
        Should not be a near-perfect copy of TSH alone.
        """
        np.random.seed(2)
        n  = 300
        df = pd.DataFrame({
            "TSH":   np.random.uniform(0.01, 50.0, n),
            "T3":    np.random.uniform(0.1,  3.0,  n),
            "label": np.random.randint(0, 3, n),
        })
        result   = engineer_thyroid(df, save=False)
        corr_tsh = result["tsh_t3_ratio"].corr(result["TSH"])
        assert abs(corr_tsh) < 0.95, \
            f"tsh_t3_ratio too correlated with TSH: r={corr_tsh:.3f}"

    def test_original_columns_preserved(self):
        df     = make_thyroid_df()
        result = engineer_thyroid(df.copy(), save=False)
        for col in df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self):
        df     = make_thyroid_df()
        result = engineer_thyroid(df.copy(), save=False)
        assert len(result) == len(df)

    def test_handles_missing_t3_column(self):
        """If T3 is absent, ratio is skipped and no error raised."""
        df     = make_thyroid_df().drop(columns=["T3"])
        result = engineer_thyroid(df, save=False)
        assert "tsh_t3_ratio" not in result.columns

    def test_handles_missing_tsh_column(self):
        """If TSH is absent, ratio is skipped and no error raised."""
        df     = make_thyroid_df().drop(columns=["TSH"])
        result = engineer_thyroid(df, save=False)
        assert "tsh_t3_ratio" not in result.columns

    def test_no_new_nulls_introduced(self):
        """Engineering should not introduce NaNs where there were none."""
        df     = make_thyroid_df()
        result = engineer_thyroid(df.copy(), save=False)
        assert result["tsh_t3_ratio"].isna().sum() == 0

    def test_accepts_dataframe_directly(self):
        """Passing a DataFrame as first arg should not attempt file read."""
        df     = make_thyroid_df()
        result = engineer_thyroid(df, save=False)
        assert isinstance(result, pd.DataFrame)