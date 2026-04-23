"""
test_models.py — unit tests for model training logic.
Run: pytest tests/test_models.py -v

Tests the helper functions without needing MLflow or a real dataset.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import json

import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed — skipping model tests")


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_ckd_train_data(n=200):
    np.random.seed(42)
    ckd = pd.DataFrame({
        "sc":             np.random.uniform(2.0, 8.0, n // 2),
        "bu":             np.random.uniform(50,  200, n // 2),
        "hemo":           np.random.uniform(6,   11,  n // 2),
        "kidney_stress":  np.random.uniform(0.5, 1.0, n // 2),
        "classification": [1] * (n // 2),
    })
    healthy = pd.DataFrame({
        "sc":             np.random.uniform(0.6, 1.2, n // 2),
        "bu":             np.random.uniform(8,   20,  n // 2),
        "hemo":           np.random.uniform(13,  17,  n // 2),
        "kidney_stress":  np.random.uniform(0.0, 0.3, n // 2),
        "classification": [0] * (n // 2),
    })
    df = pd.concat([ckd, healthy], ignore_index=True)
    return df[["sc", "bu", "hemo", "kidney_stress"]], df["classification"]


def make_thyroid_train_data(n=300):
    np.random.seed(0)
    normal = pd.DataFrame({
        "TSH":          np.random.uniform(0.5, 3.5, n // 3),
        "T3":           np.random.uniform(1.0, 2.0, n // 3),
        "tsh_t3_ratio": np.random.uniform(0.3, 3.0, n // 3),
        "label":        [0] * (n // 3),
    })
    hypo = pd.DataFrame({
        "TSH":          np.random.uniform(8.0, 40.0, n // 3),
        "T3":           np.random.uniform(0.2, 0.8,  n // 3),
        "tsh_t3_ratio": np.random.uniform(15,  80,   n // 3),
        "label":        [1] * (n // 3),
    })
    hyper = pd.DataFrame({
        "TSH":          np.random.uniform(0.01, 0.3, n // 3),
        "T3":           np.random.uniform(2.5,  6.0, n // 3),
        "tsh_t3_ratio": np.random.uniform(0.01, 0.1, n // 3),
        "label":        [2] * (n // 3),
    })
    df = pd.concat([normal, hypo, hyper], ignore_index=True)
    return df[["TSH", "T3", "tsh_t3_ratio"]], df["label"]


#  param grid builder 

class TestBuildParamGrid:
    def test_produces_correct_number_of_combos(self):
        from src.models.train_ckd import build_param_grid
        cfg = {
            "search_grid": {
                "n_estimators": [100, 200],
                "max_depth":    [3, 4],
            },
            "fixed":            {"random_state": 42},
            "max_combinations": 10,
        }
        grid = build_param_grid(cfg)
        assert len(grid) == 4   # 2×2, under max

    def test_caps_at_max_combinations(self):
        from src.models.train_ckd import build_param_grid
        cfg = {
            "search_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth":    [3, 4, 6],
                "learning_rate": [0.05, 0.1, 0.2],
            },
            "fixed":            {},
            "max_combinations": 5,
        }
        grid = build_param_grid(cfg)
        assert len(grid) == 5

    def test_fixed_params_merged_into_every_combo(self):
        from src.models.train_ckd import build_param_grid
        cfg = {
            "search_grid":      {"n_estimators": [100, 200]},
            "fixed":            {"random_state": 42, "eval_metric": "logloss"},
            "max_combinations": 10,
        }
        grid = build_param_grid(cfg)
        for combo in grid:
            assert combo["random_state"] == 42
            assert combo["eval_metric"]  == "logloss"

    def test_all_combos_have_search_keys(self):
        from src.models.train_ckd import build_param_grid
        cfg = {
            "search_grid": {
                "n_estimators": [100, 200],
                "max_depth":    [3, 4],
            },
            "fixed":            {},
            "max_combinations": 10,
        }
        grid = build_param_grid(cfg)
        for combo in grid:
            assert "n_estimators" in combo
            assert "max_depth"    in combo


#  CKD pipeline training 

class TestCKDPipeline:
    def _make_pipeline(self):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   XGBClassifier(
                n_estimators=50, max_depth=3,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42,
            )),
        ])

    def test_pipeline_fits_and_predicts(self):
        X, y   = make_ckd_train_data()
        pipe   = self._make_pipeline()
        pipe.fit(X, y)
        preds  = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_pipeline_predict_proba_shape(self):
        X, y   = make_ckd_train_data()
        pipe   = self._make_pipeline()
        pipe.fit(X, y)
        proba  = pipe.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_pipeline_handles_missing_values(self):
        X, y   = make_ckd_train_data()
        X      = X.copy()
        X.iloc[0, 0] = np.nan
        X.iloc[1, 1] = np.nan
        pipe   = self._make_pipeline()
        pipe.fit(X, y)
        preds  = pipe.predict(X)
        assert len(preds) == len(y)

    def test_auc_above_threshold_on_separable_data(self):
        from sklearn.metrics import roc_auc_score
        X, y       = make_ckd_train_data(n=300)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        pipe = self._make_pipeline()
        pipe.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        assert auc > 0.85, f"AUC too low on separable data: {auc:.3f}"


# Thyroid pipeline training

class TestThyroidPipeline:
    def _make_pipeline(self):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   XGBClassifier(
                n_estimators=50, max_depth=3,
                num_class=3, objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False, random_state=42,
            )),
        ])

    def test_pipeline_fits_and_predicts_3_classes(self):
        X, y   = make_thyroid_train_data()
        pipe   = self._make_pipeline()
        pipe.fit(X, y)
        preds  = pipe.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_pipeline_predict_proba_has_3_cols(self):
        X, y   = make_thyroid_train_data()
        pipe   = self._make_pipeline()
        pipe.fit(X, y)
        proba  = pipe.predict_proba(X)
        assert proba.shape[1] == 3
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_pipeline_handles_missing_values(self):
        X, y   = make_thyroid_train_data()
        X      = X.copy()
        X.iloc[0, 0] = np.nan
        pipe   = self._make_pipeline()
        pipe.fit(X, y)
        preds  = pipe.predict(X)
        assert len(preds) == len(y)

    def test_macro_f1_above_threshold_on_separable_data(self):
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        X, y       = make_thyroid_train_data(n=300)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        pipe = self._make_pipeline()
        pipe.fit(X_tr, y_tr)
        f1   = f1_score(y_te, pipe.predict(X_te), average="macro")
        assert f1 > 0.80, f"Macro-F1 too low on separable data: {f1:.3f}"


class TestTrainMainModelSelection:
    """Checks new --model / active_models behavior in train scripts."""

    def _patch_common_train_flow(self, monkeypatch, module, tmp_path, family: str):
        cfg_key = "ckd" if family == "ckd" else "thyroid"
        target_metric = "auc" if family == "ckd" else "macro_f1"

        cfg = {
            "experiment_name": "test-exp",
            "registered_model": "test-model",
            "dataset_version": "test-v",
            "test_size": 0.2,
            "random_state": 42,
            "data_path": "unused.csv",
            "target_col": "classification" if family == "ckd" else "label",
            "features": ["a", "b"],
            "reports_dir": str(tmp_path / "reports"),
            "model_out": str(tmp_path / "models" / f"{family}.pkl"),
            "metrics_out": str(tmp_path / "reports" / f"{family}_metrics.json"),
            "active_models": "both",
        }

        monkeypatch.setattr(module, "load_params", lambda: cfg)

        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        y = pd.Series([0, 1, 0, 1]) if family == "ckd" else pd.Series([0, 1, 2, 0])
        monkeypatch.setattr(module, "load_data", lambda _cfg: (X, y))
        monkeypatch.setattr(module, "train_test_split", lambda X, y, **k: (X, X, y, y))

        class DummyRun:
            info = type("Info", (), {"run_id": "rid"})()

        class DummyRunCtx:
            def __enter__(self):
                return DummyRun()

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(module.mlflow, "start_run", lambda **k: DummyRunCtx())
        monkeypatch.setattr(module.mlflow, "set_tracking_uri", lambda *a, **k: None)
        monkeypatch.setattr(module.mlflow, "set_experiment", lambda *a, **k: None)
        monkeypatch.setattr(module.mlflow, "log_param", lambda *a, **k: None)
        monkeypatch.setattr(module.mlflow, "log_params", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(module.mlflow, "log_metric", lambda *a, **k: None)
        monkeypatch.setattr(module.mlflow, "log_metrics", lambda *a, **k: None)
        monkeypatch.setattr(module.mlflow, "log_artifact", lambda *a, **k: None)
        monkeypatch.setattr(module.mlflow, "active_run", lambda: DummyRun())
        monkeypatch.setattr(module.mlflow.xgboost, "autolog", lambda **k: None)
        monkeypatch.setattr(module.mlflow.sklearn, "autolog", lambda **k: None)
        monkeypatch.setattr(module.mlflow, "register_model", lambda *a, **k: type("MV", (), {"version": "1"})())

        class DummyClient:
            def __init__(self, *a, **k):
                pass

            def transition_model_version_stage(self, **kwargs):
                return None

        monkeypatch.setattr(module, "MlflowClient", DummyClient)

        monkeypatch.setattr(module, "log_feature_importances", lambda *a, **k: None)
        monkeypatch.setattr(module, "log_confusion_matrix", lambda *a, **k: None)
        monkeypatch.setattr(module, "log_classification_report", lambda *a, **k: None)

        calls = {"xgb": 0, "rf": 0}

        def fake_xgb(*args, **kwargs):
            calls["xgb"] += 1
            return [
                {
                    "accuracy": 0.9,
                    target_metric: 0.91,
                    "model_type": "XGBoost",
                    "run_name": "xgb_run",
                    "run_id": "xgb123",
                    "pipeline": {"name": "xgb_pipe"},
                }
            ]

        def fake_rf(*args, **kwargs):
            calls["rf"] += 1
            return [
                {
                    "accuracy": 0.88,
                    target_metric: 0.89,
                    "model_type": "RandomForest",
                    "run_name": "rf_run",
                    "run_id": "rf123",
                    "pipeline": {"name": "rf_pipe"},
                }
            ]

        monkeypatch.setattr(module, "run_xgboost_experiments", fake_xgb)
        monkeypatch.setattr(module, "run_rf_experiments", fake_rf)
        return cfg, calls

    def test_ckd_main_uses_active_models_both(self, monkeypatch, tmp_path):
        from src.models import train_ckd as mod

        cfg, calls = self._patch_common_train_flow(monkeypatch, mod, tmp_path, family="ckd")
        mod.main(model_override=None)

        assert calls["xgb"] == 1
        assert calls["rf"] == 1
        with open(cfg["metrics_out"]) as fh:
            metrics = json.load(fh)
        assert metrics["run_name"] == "xgb_run"
        assert "auc" in metrics

    def test_ckd_main_model_override_rf(self, monkeypatch, tmp_path):
        from src.models import train_ckd as mod

        _, calls = self._patch_common_train_flow(monkeypatch, mod, tmp_path, family="ckd")
        mod.main(model_override="rf")
        assert calls["xgb"] == 0
        assert calls["rf"] == 1

    def test_thyroid_main_model_override_xgb(self, monkeypatch, tmp_path):
        from src.models import train_thyroid as mod

        _, calls = self._patch_common_train_flow(monkeypatch, mod, tmp_path, family="thyroid")
        mod.main(model_override="xgb")
        assert calls["xgb"] == 1
        assert calls["rf"] == 0
