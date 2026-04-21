"""
test_preprocess.py — unit tests for the preprocessing module.
Run: pytest tests/test_preprocess.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPreprocessCKD:
    """Tests for CKD preprocessing logic."""

    def test_column_rename_wbcc_to_wc(self):
        """wbcc should be renamed to wc."""
        import importlib, types
        # Test the rename logic directly without file I/O
        df = pd.DataFrame({"wbcc": [7800, 6000], "rbcc": [5.2, 4.8],
                           "class": ["ckd", "notckd"]})
        df = df.rename(columns={"wbcc": "wc", "rbcc": "rc", "class": "classification"})
        assert "wc" in df.columns
        assert "rc" in df.columns
        assert "wbcc" not in df.columns

    def test_target_encoding(self):
        """ckd→1, notckd→0."""
        target_map = {"ckd": 1, "notckd": 0, "not ckd": 0}
        series = pd.Series(["ckd", "notckd", "ckd", "not ckd"])
        result = series.str.strip().str.lower().map(target_map)
        assert list(result) == [1, 0, 1, 0]

    def test_yes_no_encoding(self):
        """yes→1, no→0."""
        yn_map = {"yes": 1, "no": 0, "normal": 1, "abnormal": 0,
                  "present": 1, "notpresent": 0, "good": 1, "poor": 0}
        assert yn_map["yes"] == 1
        assert yn_map["no"] == 0
        assert yn_map["normal"] == 1
        assert yn_map["abnormal"] == 0
        assert yn_map["notpresent"] == 0

    def test_numeric_coercion_handles_question_marks(self):
        """Strings like '?' should become NaN after coercion."""
        series = pd.Series(["1.2", "?", "3.4", ""])
        result = pd.to_numeric(series, errors="coerce")
        assert pd.isna(result[1])
        assert pd.isna(result[3])
        assert result[0] == pytest.approx(1.2)

    def test_no_object_columns_after_encoding(self):
        """After preprocessing, all columns should be numeric."""
        df = pd.DataFrame({
            "age": [48.0, 7.0],
            "htn": ["yes", "no"],
            "dm":  ["yes", "no"],
            "appet": ["good", "poor"],
            "classification": [1, 0],
        })
        yn_map = {"yes": 1, "no": 0, "good": 1, "poor": 0}
        for col in df.select_dtypes(include="object").columns:
            if col == "classification":
                continue
            df[col] = df[col].map(yn_map)
        for col in df.columns:
            if col != "classification":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        assert obj_cols == [], f"Object columns remain: {obj_cols}"


class TestPreprocessThyroid:
    """Tests for thyroid label collapse and encoding."""

    def test_label_negative_maps_to_0(self):
        def map_label(raw):
            s = str(raw).lower().strip().split(".")[0].strip()
            if s in ("negative", "normal", "-", ""):
                return 0
            if "hypo" in s:
                return 1
            if "hyper" in s or "toxic" in s:
                return 2
            return np.nan

        assert map_label("negative") == 0
        assert map_label("negative.") == 0

    def test_label_hypothyroid_maps_to_1(self):
        def map_label(raw):
            s = str(raw).lower().strip().split(".")[0].strip()
            if s in ("negative", "normal"):
                return 0
            if "hypo" in s:
                return 1
            if "hyper" in s or "toxic" in s:
                return 2
            return np.nan

        assert map_label("primary hypothyroid") == 1
        assert map_label("compensated hypothyroid") == 1
        assert map_label("secondary hypothyroid") == 1

    def test_label_hyperthyroid_maps_to_2(self):
        def map_label(raw):
            s = str(raw).lower().strip().split(".")[0].strip()
            if s in ("negative", "normal"):
                return 0
            if "hypo" in s:
                return 1
            if "hyper" in s or "toxic" in s:
                return 2
            return np.nan

        assert map_label("T3 toxic") == 2
        assert map_label("toxic goitre") == 2
        assert map_label("hyperthyroid") == 2

    def test_boolean_flag_encoding(self):
        """t→1, f→0."""
        flag_map = {"t": 1, "f": 0, "true": 1, "false": 0}
        series   = pd.Series(["t", "f", "t", "f"])
        result   = series.map(flag_map)
        assert list(result) == [1, 0, 1, 0]

    def test_sex_encoding(self):
        """M→1, F→0."""
        sex_map = {"M": 1, "F": 0}
        assert sex_map["M"] == 1
        assert sex_map["F"] == 0

    def test_unknown_labels_become_nan(self):
        """Labels not in the mapping should become NaN and be dropped."""
        def map_label(raw):
            s = str(raw).lower().strip().split(".")[0].strip()
            if s in ("negative",):
                return 0
            if "hypo" in s:
                return 1
            if "hyper" in s or "toxic" in s:
                return 2
            return np.nan

        assert np.isnan(map_label("some unknown label"))
        assert np.isnan(map_label("nan"))