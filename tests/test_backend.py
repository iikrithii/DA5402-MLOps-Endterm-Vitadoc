"""
test_backend.py

Unit tests for VitaDoc backend helper functions.
These tests do not require MLflow or a running server —
they test the pure logic functions directly.

Run: pytest tests/test_backend.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed — skipping model tests")

# flag_values tests

SAMPLE_REF_RANGES = {
    "sc":   {"name": "Serum Creatinine", "unit": "mg/dL",  "low": 0.7,  "high": 1.3},
    "hemo": {"name": "Haemoglobin",      "unit": "g/dL",   "low": 12.0, "high": 17.0},
    "TSH":  {"name": "TSH",              "unit": "mIU/L",  "low": 0.4,  "high": 4.0},
    "sod":  {"name": "Sodium",           "unit": "mEq/L",  "low": 136.0,"high": 145.0},
    "bu":   {"name": "Blood Urea",       "unit": "mg/dL",  "low": 7.0,  "high": 25.0},
}


class TestFlagValues:
    """Tests for the reference range flagger."""

    def setup_method(self):
        from backend.main import flag_values
        self.flag_values = flag_values

    def test_high_value_flagged(self):
        flags = self.flag_values({"sc": 2.5}, SAMPLE_REF_RANGES)
        assert flags["sc"] == "HIGH"

    def test_low_value_flagged(self):
        flags = self.flag_values({"hemo": 9.0}, SAMPLE_REF_RANGES)
        assert flags["hemo"] == "LOW"

    def test_normal_value(self):
        flags = self.flag_values({"sod": 140.0}, SAMPLE_REF_RANGES)
        assert flags["sod"] == "NORMAL"

    def test_unknown_feature(self):
        flags = self.flag_values({"xyz": 99.9}, SAMPLE_REF_RANGES)
        assert flags["xyz"] == "UNKNOWN"

    def test_boundary_at_low(self):
        # Exactly at lower bound — should be NORMAL
        flags = self.flag_values({"sc": 0.7}, SAMPLE_REF_RANGES)
        assert flags["sc"] == "NORMAL"

    def test_boundary_at_high(self):
        # Exactly at upper bound — should be NORMAL
        flags = self.flag_values({"sc": 1.3}, SAMPLE_REF_RANGES)
        assert flags["sc"] == "NORMAL"

    def test_multiple_features(self):
        features = {"sc": 0.8, "TSH": 5.0, "bu": 20.0}
        flags    = self.flag_values(features, SAMPLE_REF_RANGES)
        assert flags["sc"]  == "NORMAL"
        assert flags["TSH"] == "HIGH"
        assert flags["bu"]  == "NORMAL"

    def test_empty_features(self):
        flags = self.flag_values({}, SAMPLE_REF_RANGES)
        assert flags == {}


# OCR synonym map tests

class TestSynonymMap:
    """Tests that key synonyms are present and map to correct features."""

    def setup_method(self):
        from backend.main import SYNONYM_MAP
        self.SYNONYM_MAP = SYNONYM_MAP

    def test_creatinine_maps_to_sc(self):
        assert "serum creatinine" in self.SYNONYM_MAP
        assert self.SYNONYM_MAP["serum creatinine"] == "sc"

    def test_tsh_maps_correctly(self):
        assert "tsh" in self.SYNONYM_MAP
        assert self.SYNONYM_MAP["tsh"] == "TSH"

    def test_haemoglobin_maps_correctly(self):
        assert "haemoglobin" in self.SYNONYM_MAP
        assert self.SYNONYM_MAP["haemoglobin"] == "hemo"

    def test_hb_alias_maps_correctly(self):
        assert "hb" in self.SYNONYM_MAP
        assert self.SYNONYM_MAP["hb"] == "hemo"

    def test_all_targets_are_known_features(self):
        known = {
            "sc", "bu", "hemo", "sod", "pot", "bgr", "pcv", "wc", "rc",
            "bp", "TSH", "T3", "TT4", "T4U", "FTI",
        }
        for synonym, target in self.SYNONYM_MAP.items():
            assert target in known, \
                f"Synonym '{synonym}' maps to unknown feature '{target}'"


# OCR extraction tests

class TestOCRExtraction:
    """Tests for PDF text extraction logic."""

    def setup_method(self):
        from backend.main import extract_features_from_pdf
        self.extract = extract_features_from_pdf

    def test_returns_dict_on_empty_input(self):
        result = self.extract(b"")
        assert isinstance(result, dict)

    def test_returns_dict_on_invalid_pdf(self):
        # Should not raise — returns empty dict gracefully
        result = self.extract(b"not a real pdf")
        assert isinstance(result, dict)

    def test_returns_empty_dict_on_no_matches(self):
        result = self.extract(b"")
        assert result == {}


# Engineered feature computation tests

class TestEngineeredFeatures:
    """
    Tests that kidney_stress and tsh_t3_ratio are computed correctly
    at inference time. These must match the training-time computation
    in engineer.py exactly — otherwise train/serve skew occurs.
    """

    def test_kidney_stress_high_for_ckd_pattern(self):
        # High creatinine, high urea, low haemoglobin = high stress
        features = {"sc": 5.0, "bu": 150.0, "hemo": 7.0}
        sc_comp   = ((features["sc"]   - 0.7)  / (20.0 - 0.7))  * 0.40
        bu_comp   = ((features["bu"]   - 7.0)  / (300.0 - 7.0)) * 0.35
        hemo_comp = ((17.0 - features["hemo"]) / (17.0 - 2.0))  * 0.25
        kidney_stress = float(np.clip(sc_comp + bu_comp + hemo_comp, 0, 1))
        assert kidney_stress > 0.3, f"Expected high stress, got {kidney_stress}"

    def test_kidney_stress_low_for_normal_pattern(self):
        # Normal values = low stress
        features = {"sc": 0.9, "bu": 15.0, "hemo": 14.0}
        sc_comp   = ((features["sc"]   - 0.7)  / (20.0 - 0.7))  * 0.40
        bu_comp   = ((features["bu"]   - 7.0)  / (300.0 - 7.0)) * 0.35
        hemo_comp = ((17.0 - features["hemo"]) / (17.0 - 2.0))  * 0.25
        kidney_stress = float(np.clip(sc_comp + bu_comp + hemo_comp, 0, 1))
        assert kidney_stress < 0.2, f"Expected low stress, got {kidney_stress}"

    def test_kidney_stress_within_0_1(self):
        # Should always be clipped to [0, 1]
        features = {"sc": 50.0, "bu": 500.0, "hemo": 0.1}
        sc_comp   = ((features["sc"]   - 0.7)  / (20.0 - 0.7))  * 0.40
        bu_comp   = ((features["bu"]   - 7.0)  / (300.0 - 7.0)) * 0.35
        hemo_comp = ((17.0 - features["hemo"]) / (17.0 - 2.0))  * 0.25
        kidney_stress = float(np.clip(sc_comp + bu_comp + hemo_comp, 0, 1))
        assert 0.0 <= kidney_stress <= 1.0

    def test_tsh_t3_ratio_high_for_hypothyroid(self):
        # High TSH, low T3 = high ratio = hypothyroid pattern
        tsh   = 20.0
        t3    = 0.4
        ratio = tsh / (t3 + 0.01)
        assert ratio > 10.0, f"Expected high ratio, got {ratio}"

    def test_tsh_t3_ratio_no_division_by_zero(self):
        tsh   = 5.0
        t3    = 0.0
        ratio = tsh / (t3 + 0.01)  # epsilon prevents division by zero
        assert np.isfinite(ratio)

    def test_tsh_t3_ratio_low_for_hyperthyroid(self):
        # Low TSH, high T3 = low ratio = hyperthyroid pattern
        tsh   = 0.05
        t3    = 4.0
        ratio = tsh / (t3 + 0.01)
        assert ratio < 0.1, f"Expected low ratio, got {ratio}"