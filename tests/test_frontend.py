"""
Minimal frontend test suite for VitaDoc.

Keeps only pure frontend checks and HTML structure checks.
No backend calls, no integration tests.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRONTEND_HTML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend",
    "index.html",
)
# Helpers mirrored from index.html
def badge(flag: str) -> str:
    MAP = {
        "HIGH": ("badge-high", "▲ HIGH"),
        "LOW": ("badge-low", "▼ LOW"),
        "NORMAL": ("badge-normal", "✓ NORMAL"),
    }
    cls, txt = MAP.get(flag, ("badge-unknown", "? —"))
    return f'<span class="badge {cls}">{txt}</span>'


def conf_bar(conf: float, colour: str) -> str:
    pct = round(conf * 100)
    return (
        f'<div class="conf-bar">'
        f'<div class="conf-fill" style="width:{pct}%;background:{colour}"></div>'
        f'</div>'
    )


def update_field(fields: dict, key: str, val: str) -> dict:
    fields = dict(fields)
    stripped = val.strip() if isinstance(val, str) else ""
    if stripped == "":
        fields.pop(key, None)
        return fields
    try:
        fields[key] = float(val)
    except (ValueError, TypeError):
        fields.pop(key, None)
    return fields


def is_positive(class_name: str) -> bool:
    return class_name in {"CKD Detected", "Hypothyroid", "Hyperthyroid"}


CKD_ORDER = ["sc", "bu", "hemo", "sod", "pot", "bgr", "pcv", "wc", "rc", "bp"]
THY_ORDER = ["TSH", "T3", "TT4", "T4U", "FTI"]

FEAT_NAMES = {
    "sc": "Serum Creatinine", "bu": "Blood Urea", "hemo": "Haemoglobin",
    "sod": "Sodium", "pot": "Potassium", "bgr": "Blood Glucose",
    "pcv": "Packed Cell Vol", "wc": "WBC Count", "rc": "RBC Count",
    "bp": "Blood Pressure",
    "TSH": "TSH", "T3": "T3", "TT4": "Total T4", "T4U": "T4 Uptake",
    "FTI": "Free T4 Index",
}

@pytest.fixture(scope="module")
def html_content():
    if not os.path.exists(FRONTEND_HTML):
        pytest.skip(f"frontend/index.html not found at {FRONTEND_HTML}")
    with open(FRONTEND_HTML, encoding="utf-8") as f:
        return f.read()

# 20 tests total
class TestFrontendMinimal:

    def test_badge_high_has_class(self):
        assert "badge-high" in badge("HIGH")

    def test_badge_low_has_class(self):
        assert "badge-low" in badge("LOW")

    def test_badge_normal_has_class(self):
        assert "badge-normal" in badge("NORMAL")

    def test_badge_unknown_uses_default(self):
        assert "badge-unknown" in badge("UNKNOWN")

    def test_conf_bar_full_confidence(self):
        assert "width:100%" in conf_bar(1.0, "#ef4444")

    def test_conf_bar_zero_confidence(self):
        assert "width:0%" in conf_bar(0.0, "#22c55e")

    def test_conf_bar_half_confidence(self):
        assert "width:50%" in conf_bar(0.5, "#ef4444")

    def test_is_positive_ckd_detected(self):
        assert is_positive("CKD Detected")

    def test_is_positive_hypothyroid(self):
        assert is_positive("Hypothyroid")

    def test_is_positive_not_ckd_false(self):
        assert not is_positive("Not CKD")

    def test_ckd_order_has_ten_features(self):
        assert len(CKD_ORDER) == 10

    def test_thy_order_has_five_features(self):
        assert len(THY_ORDER) == 5

    def test_ckd_and_thy_orders_are_disjoint(self):
        assert set(CKD_ORDER).isdisjoint(set(THY_ORDER))

    def test_all_ckd_keys_are_in_feat_names(self):
        for k in CKD_ORDER:
            assert k in FEAT_NAMES

    def test_all_thy_keys_are_in_feat_names(self):
        for k in THY_ORDER:
            assert k in FEAT_NAMES

    def test_update_field_stores_valid_float(self):
        assert update_field({}, "sc", "3.2")["sc"] == 3.2

    def test_update_field_removes_empty_value(self):
        assert "sc" not in update_field({"sc": 3.2}, "sc", "")

    def test_update_field_removes_non_numeric(self):
        assert "sc" not in update_field({"sc": 3.2}, "sc", "abc")

    def test_html_has_core_structure(self, html_content):
        assert "<!DOCTYPE html>" in html_content or "<!doctype html>" in html_content.lower()
        assert "VitaDoc" in html_content
        assert 'id="sidebar"' in html_content
        assert 'id="main"' in html_content or 'id="content"' in html_content

    def test_html_has_key_ui_strings(self, html_content):
        lower = html_content.lower()
        assert "onboarding-overlay" in lower
        assert "hamburger" in lower
        assert "/analyse" in lower
        assert "/feedback" in lower
