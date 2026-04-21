"""
train_thyroid.py

Trains thyroid dysfunction classifiers (3-class: Normal, Hypothyroid,
Hyperthyroid). Supports XGBoost, Random Forest, or both.

SMOTE is particularly important here because Hyperthyroid is only
2.2% of the dataset. Without balancing, the model learns to ignore
that class entirely and macro-F1 drops significantly.

Usage:
    python src/models/train_thyroid.py                  # uses params.yaml
    python src/models/train_thyroid.py --model xgb
    python src/models/train_thyroid.py --model rf
    python src/models/train_thyroid.py --model both
"""

import argparse
import json
import logging
import os
import pickle
import time
from itertools import product
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PARAMS_PATH = "params.yaml"
CLASS_NAMES = ["Normal", "Hypothyroid", "Hyperthyroid"]


def load_params() -> Dict:
    """Load the thyroid section from params.yaml."""
    with open(PARAMS_PATH) as fh:
        return yaml.safe_load(fh)["thyroid"]


def build_param_grid(grid_cfg: Dict) -> List[Dict]:
    """
    Expand a search_grid block from params.yaml into a flat list of
    parameter dicts. Fixed params are merged into every combination.

    Caps at max_combinations using evenly-spaced sampling so the
    full hyperparameter range is covered even when the grid is large.

    Args:
        grid_cfg: one of cfg['xgb'] or cfg['rf'] from params.yaml.

    Returns:
        List of param dicts ready to pass to a model constructor.
    """
    grid  = grid_cfg["search_grid"]
    fixed = grid_cfg.get("fixed", {})
    max_n = grid_cfg.get("max_combinations", 12)

    keys   = list(grid.keys())
    values = list(grid.values())
    combos = [dict(zip(keys, combo)) for combo in product(*values)]

    if len(combos) > max_n:
        indices = [int(i * (len(combos) - 1) / (max_n - 1)) for i in range(max_n)]
        combos  = [combos[i] for i in indices]

    return [{**fixed, **c} for c in combos]


def load_data(cfg: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the engineered thyroid dataset from the path in params.yaml.
    Only keeps features listed in cfg['features'].

    Args:
        cfg: the 'thyroid' block from params.yaml.

    Returns:
        X: feature DataFrame
        y: label Series (0=Normal, 1=Hypo, 2=Hyper)
    """
    df        = pd.read_csv(cfg["data_path"])
    df        = df.dropna(subset=[cfg["target_col"]])
    available = [c for c in cfg["features"] if c in df.columns]
    missing   = [c for c in cfg["features"] if c not in df.columns]

    if missing:
        log.warning("Features not found (skipped): %s", missing)

    X = df[available]
    y = df[cfg["target_col"]].astype(int)

    log.info(
        "Thyroid loaded | rows=%d  features=%d  dist=%s",
        len(df), len(available), dict(y.value_counts()),
    )
    return X, y


def log_confusion_matrix(y_test, y_pred, run_name: str, reports_dir: str):
    """
    Save a 3-class confusion matrix PNG and log it as an MLflow artifact.

    Args:
        run_name:    used in the filename to avoid overwriting between runs.
        reports_dir: output directory from params.yaml.
    """
    os.makedirs(reports_dir, exist_ok=True)
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Greens")
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title(f"Thyroid — {run_name}")
    plt.tight_layout()
    path = os.path.join(reports_dir, f"thyroid_cm_{run_name}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    mlflow.log_artifact(path)


def log_classification_report(y_test, y_pred, run_name: str, reports_dir: str):
    """
    Save a text classification report and log it as an MLflow artifact.

    Args:
        reports_dir: output directory from params.yaml.
    """
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"thyroid_report_{run_name}.txt")
    with open(path, "w") as fh:
        fh.write(
            classification_report(y_test, y_pred, target_names=CLASS_NAMES)
        )
    mlflow.log_artifact(path)


def log_feature_importances(pipeline: Pipeline, feature_names: List[str]):
    """
    Log each feature's importance as an individual MLflow metric.
    Works for both XGBoost and RandomForest.

    Also logs the top 3 as a summary param visible in the run list
    without having to click into each individual run.

    Args:
        pipeline:      fitted sklearn Pipeline.
        feature_names: column names in the order passed to fit.
    """
    importances = pipeline.named_steps["model"].feature_importances_
    for feat, imp in zip(feature_names, importances):
        mlflow.log_metric(f"importance_{feat}", float(imp))

    top3 = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1], reverse=True,
    )[:3]
    mlflow.log_param(
        "top3_features",
        ", ".join(f"{f}({v:.3f})" for f, v in top3),
    )
    log.info("Top 3 features: %s", top3)


def run_one_experiment(
    run_name:    str,
    X_train:     pd.DataFrame,
    X_test:      pd.DataFrame,
    y_train:     pd.Series,
    y_test:      pd.Series,
    model,
    model_type:  str,
    params:      Dict,
    cfg:         Dict,
    apply_smote: bool = False,
) -> Dict:
    """
    Run one thyroid training experiment and log everything to MLflow.

    Uses macro_f1 as the primary comparison metric rather than AUC
    because this is a 3-class problem with severe class imbalance.
    Macro-F1 weights all three classes equally, which penalises models
    that ignore the minority Hyperthyroid class.

    Args:
        model:       unfitted model instance.
        model_type:  "XGBoost" or "RandomForest", logged for UI filtering.
        params:      hyperparameters for this run.
        cfg:         thyroid config block from params.yaml.
        apply_smote: whether to apply SMOTE before training.

    Returns:
        Dict with accuracy, macro_f1, model_type, run_name, run_id, pipeline.
    """
    with mlflow.start_run(run_name=run_name):

        if model_type == "XGBoost":
            mlflow.xgboost.autolog(silent=True)
        else:
            mlflow.sklearn.autolog(silent=True)

        mlflow.log_param("model_type",      model_type)
        mlflow.log_param("dataset_version", cfg["dataset_version"])
        mlflow.log_param("train_size",      len(X_train))
        mlflow.log_param("test_size",       len(X_test))
        mlflow.log_param("feature_count",   X_train.shape[1])
        mlflow.log_param("smote_applied",   apply_smote)
        for k, v in params.items():
            mlflow.log_param(f"param_{k}", v)

        X_tr = X_train.copy()
        y_tr = y_train.copy()

        if apply_smote:
            try:
                imp      = SimpleImputer(strategy="median")
                X_tr_imp = imp.fit_transform(X_tr)
                sm       = SMOTE(random_state=cfg["random_state"])
                X_tr_imp, y_tr = sm.fit_resample(X_tr_imp, y_tr)
                X_tr     = pd.DataFrame(X_tr_imp, columns=X_train.columns)
                mlflow.log_param(
                    "smote_ratio",
                    dict(pd.Series(y_tr).value_counts()),
                )
            except Exception as e:
                log.error("SMOTE failed: %s — continuing without it", e)
                X_tr = X_train.copy()
                y_tr = y_train.copy()

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   model),
        ])

        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        train_time = time.time() - t0

        y_pred       = pipe.predict(X_test)
        acc          = accuracy_score(y_test, y_pred)
        macro_f1     = f1_score(y_test, y_pred, average="macro")
        per_class_f1 = f1_score(y_test, y_pred, average=None)

        mlflow.log_metrics({
            "accuracy":     acc,
            "macro_f1":     macro_f1,
            "f1_normal":    float(per_class_f1[0]),
            "f1_hypo":      float(per_class_f1[1]),
            "f1_hyper":     float(per_class_f1[2]) if len(per_class_f1) > 2 else 0.0,
            "train_time_s": train_time,
        })

        log_feature_importances(pipe, X_test.columns.tolist())
        log_confusion_matrix(y_test, y_pred, run_name, cfg["reports_dir"])
        log_classification_report(y_test, y_pred, run_name, cfg["reports_dir"])

        run_id = mlflow.active_run().info.run_id
        log.info(
            "%-50s  acc=%.4f  macro_f1=%.4f  t=%.1fs",
            run_name, acc, macro_f1, train_time,
        )

        return {
            "accuracy":   acc,
            "macro_f1":   macro_f1,
            "model_type": model_type,
            "run_name":   run_name,
            "run_id":     run_id,
            "pipeline":   pipe,
        }


def run_xgboost_experiments(X_train, X_test, y_train, y_test, cfg) -> List[Dict]:
    """
    Run the full XGBoost experiment set for thyroid classification.

    Goes through every grid combination in cfg['xgb']['search_grid'],
    then runs the best combination again with and without SMOTE.

    Args:
        cfg: thyroid config block from params.yaml.

    Returns:
        List of result dicts, one per MLflow run.
    """
    combos = build_param_grid(cfg["xgb"])
    log.info("XGBoost Thyroid grid: %d combinations", len(combos))
    results = []

    for params in combos:
        run_name = (
            f"thy_xgb"
            f"_ne{params['n_estimators']}"
            f"_d{params['max_depth']}"
            f"_lr{params['learning_rate']}"
        )
        results.append(run_one_experiment(
            run_name, X_train, X_test, y_train, y_test,
            XGBClassifier(**params), "XGBoost", params, cfg,
        ))

    best_p = combos[results.index(max(results, key=lambda r: r["macro_f1"]))]

    results.append(run_one_experiment(
        f"thy_xgb_smote_ne{best_p['n_estimators']}_d{best_p['max_depth']}",
        X_train, X_test, y_train, y_test,
        XGBClassifier(**best_p), "XGBoost", best_p, cfg,
        apply_smote=True,
    ))
    results.append(run_one_experiment(
        f"thy_xgb_best_ne{best_p['n_estimators']}_d{best_p['max_depth']}",
        X_train, X_test, y_train, y_test,
        XGBClassifier(**best_p), "XGBoost", best_p, cfg,
        apply_smote=False,
    ))

    return results


def run_rf_experiments(X_train, X_test, y_train, y_test, cfg) -> List[Dict]:
    """
    Run the full Random Forest experiment set for thyroid classification.

    Same structure as run_xgboost_experiments.

    Args:
        cfg: thyroid config block from params.yaml.

    Returns:
        List of result dicts, one per MLflow run.
    """
    combos = build_param_grid(cfg["rf"])
    log.info("RandomForest Thyroid grid: %d combinations", len(combos))
    results = []

    for params in combos:
        run_name = (
            f"thy_rf"
            f"_ne{params['n_estimators']}"
            f"_d{params['max_depth']}"
        )
        results.append(run_one_experiment(
            run_name, X_train, X_test, y_train, y_test,
            RandomForestClassifier(**params), "RandomForest", params, cfg,
        ))

    best_p = combos[results.index(max(results, key=lambda r: r["macro_f1"]))]

    results.append(run_one_experiment(
        f"thy_rf_smote_ne{best_p['n_estimators']}_d{best_p['max_depth']}",
        X_train, X_test, y_train, y_test,
        RandomForestClassifier(**best_p), "RandomForest", best_p, cfg,
        apply_smote=True,
    ))
    results.append(run_one_experiment(
        f"thy_rf_best_ne{best_p['n_estimators']}_d{best_p['max_depth']}",
        X_train, X_test, y_train, y_test,
        RandomForestClassifier(**best_p), "RandomForest", best_p, cfg,
        apply_smote=False,
    ))

    return results


def main(model_override: str = None):
    """
    Entry point for thyroid training.

    model_override comes from the --model CLI argument and takes
    precedence over active_models in params.yaml. This lets DVC
    pass the value from params.yaml while still allowing manual
    overrides during development.

    Args:
        model_override: "xgb", "rf", or "both". If None, reads from params.yaml.
    """
    cfg    = load_params()
    active = model_override or cfg.get("active_models", "both")

    log.info("Thyroid training | active_models=%s", active)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["experiment_name"])

    X, y = load_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )

    xgb_results = (
        run_xgboost_experiments(X_train, X_test, y_train, y_test, cfg)
        if active in ("xgb", "both") else []
    )
    rf_results = (
        run_rf_experiments(X_train, X_test, y_train, y_test, cfg)
        if active in ("rf", "both") else []
    )

    all_results = xgb_results + rf_results
    if not all_results:
        raise ValueError(f"No results — check active_models value: '{active}'")

    best = max(all_results, key=lambda r: r["macro_f1"])
    log.info(
        "Best Thyroid run: %s [%s]  macro_f1=%.4f",
        best["run_name"], best["model_type"], best["macro_f1"],
    )

    try:
        os.makedirs(os.path.dirname(cfg["model_out"]), exist_ok=True)
        with open(cfg["model_out"], "wb") as fh:
            pickle.dump(best["pipeline"], fh)
        log.info("Saved best pipeline → %s", cfg["model_out"])
    except Exception as e:
        log.error("Failed to save model: %s", e)
        raise

    os.makedirs(cfg["reports_dir"], exist_ok=True)
    best_for_json = {k: v for k, v in best.items() if k != "pipeline"}
    with open(cfg["metrics_out"], "w") as fh:
        json.dump(best_for_json, fh, indent=2)

    try:
        client    = MlflowClient(MLFLOW_URI)
        model_uri = f"runs:/{best['run_id']}/model"
        mv        = mlflow.register_model(model_uri, cfg["registered_model"])
        client.transition_model_version_stage(
            name=cfg["registered_model"],
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        log.info(
            "Registered %s v%s → Production",
            cfg["registered_model"], mv.version,
        )
    except Exception as e:
        log.error(
            "Registration failed — model saved at %s. Error: %s",
            cfg["model_out"], e,
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train thyroid classifiers")
    parser.add_argument(
        "--model",
        choices=["xgb", "rf", "both"],
        default=None,
        help="Model family to train. Overrides active_models in params.yaml.",
    )
    args = parser.parse_args()
    main(model_override=args.model)