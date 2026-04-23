"""
vitadoc_retraining.py

Nightly retraining DAG — runs at 2am.

Checks two independent conditions:
    1. Data drift   — rolling mean of key features has shifted more
                      than 2 std deviations from the training baseline
    2. User feedback — 20+ unused corrections AND correction rate > 15%

If either condition triggers, the DAG:
    1. Appends corrected samples to the raw CSVs
    2. Marks those feedback rows as used_in_training = 1
    3. Runs `dvc repro` from the project root

DVC then detects the CSV hash has changed and automatically
re-runs only the affected stages:
    validate → preprocess → engineer → baselines → train_ckd/thyroid

The new model version appears in MLflow registry automatically
because train_ckd.py and train_thyroid.py call mlflow.register_model.

No pools on this DAG — it runs a small number of sequential tasks
and does not compete with the data pipeline for resources.

Shares the same raw CSVs and DVC pipeline as vitadoc_data_pipeline.
No logic is duplicated between the two DAGs.
"""

import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

log     = logging.getLogger(__name__)
PYTHON  = sys.executable
WORKDIR = "/opt/airflow"

# Paths — use environment variable so they work both inside and outside Docker
DB_PATH       = os.getenv("VITADOC_DB_PATH", "/opt/airflow/logs/vitadoc.db")
BASELINE_PATH = os.path.join(WORKDIR, "data", "baseline_stats.json")
CKD_RAW_PATH  = os.path.join(WORKDIR, "data", "raw", "ckd.csv")
THY_RAW_PATH  = os.path.join(WORKDIR, "data", "raw", "thyroid.csv")

DRIFT_THRESHOLD    = 2.0
FEEDBACK_MIN_COUNT = 20
FEEDBACK_MIN_RATE  = 0.15

DEFAULT_ARGS = {
    "owner":            "vitadoc",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


def _db_exists() -> bool:
    """Check the DB file exists and has the expected tables."""
    if not os.path.exists(DB_PATH):
        log.info("DB not found at %s — skipping", DB_PATH)
        return False
    try:
        conn   = sqlite3.connect(DB_PATH)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        return "predictions" in tables and "feedback" in tables
    except Exception as e:
        log.warning("DB check failed: %s", e)
        return False


def check_drift(**ctx):
    """
    Check whether key features have drifted beyond the threshold.

    Reads the last 200 predictions from SQLite, computes rolling means
    per feature, and compares against baseline_stats.json.

    Pushes drift_detected (bool) to XCom.
    If fewer than 20 predictions exist, drift check is skipped.
    """
    if not _db_exists():
        ctx["ti"].xcom_push(key="drift_detected", value=False)
        return

    if not os.path.exists(BASELINE_PATH):
        log.warning("No baseline_stats.json at %s", BASELINE_PATH)
        ctx["ti"].xcom_push(key="drift_detected", value=False)
        return

    with open(BASELINE_PATH) as fh:
        baseline = json.load(fh)

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT features_json FROM predictions "
        "ORDER BY timestamp DESC LIMIT 200"
    ).fetchall()
    conn.close()

    if len(rows) < 20:
        log.info(
            "Only %d predictions — need 20 for drift check. Skipping.",
            len(rows),
        )
        ctx["ti"].xcom_push(key="drift_detected", value=False)
        return

    # Parse features from recent predictions into per-feature lists
    feature_values: dict = {}
    for (features_json,) in rows:
        try:
            for k, v in json.loads(features_json).items():
                feature_values.setdefault(k, []).append(v)
        except Exception:
            continue

    drift_detected = False
    monitored      = ["sc", "TSH", "hemo", "bu"]

    for feat in monitored:
        if feat not in feature_values or len(feature_values[feat]) < 10:
            continue

        rolling_mean = float(np.mean(feature_values[feat]))

        # Check against both CKD and thyroid baselines
        for dataset_name, dataset in baseline.items():
            if feat not in dataset:
                continue
            bstats        = dataset[feat]
            baseline_mean = bstats.get("mean", rolling_mean)
            baseline_std  = bstats.get("std",  1.0) or 1.0
            drift_stddevs = abs(rolling_mean - baseline_mean) / baseline_std

            log.info(
                "Drift %s.%s: rolling=%.3f baseline=%.3f drift=%.2f σ",
                dataset_name, feat,
                rolling_mean, baseline_mean, drift_stddevs,
            )

            if drift_stddevs > DRIFT_THRESHOLD:
                log.warning(
                    "DRIFT DETECTED: %s.%s = %.2f σ from baseline",
                    dataset_name, feat, drift_stddevs,
                )
                drift_detected = True

    ctx["ti"].xcom_push(key="drift_detected", value=drift_detected)
    log.info("Drift check complete. drift_detected=%s", drift_detected)


def check_feedback(**ctx):
    """
    Check whether enough user feedback has accumulated for retraining.

    Both conditions must be true:
        - At least FEEDBACK_MIN_COUNT unused corrections in the DB
        - Correction rate > FEEDBACK_MIN_RATE (corrections / total predictions)

    Pushes feedback_triggered (bool) and feedback_data (list) to XCom.
    feedback_data is a list of [condition, correct_label, features_json].
    """
    if not _db_exists():
        ctx["ti"].xcom_push(key="feedback_triggered", value=False)
        ctx["ti"].xcom_push(key="feedback_data",      value=[])
        return

    conn = sqlite3.connect(DB_PATH)

    feedback_rows = conn.execute(
        """SELECT f.condition, f.correct_label, p.features_json
           FROM feedback f
           JOIN predictions p ON f.prediction_id = p.id
           WHERE f.used_in_training = 0"""
    ).fetchall()

    total_preds = conn.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    conn.close()

    n_feedback = len(feedback_rows)
    rate       = n_feedback / max(total_preds, 1)

    log.info(
        "Feedback: %d corrections, %d total predictions, rate=%.1f%%",
        n_feedback, total_preds, rate * 100,
    )

    triggered = (
        n_feedback >= FEEDBACK_MIN_COUNT
        and rate    >= FEEDBACK_MIN_RATE
    )

    ctx["ti"].xcom_push(key="feedback_triggered", value=triggered)
    ctx["ti"].xcom_push(
        key="feedback_data",
        value=[[r[0], r[1], r[2]] for r in feedback_rows],
    )
    log.info("Feedback check complete. triggered=%s", triggered)


def inject_and_retrain(**ctx):
    """
    Inject feedback samples into raw CSVs and run dvc repro.

    Only executes if drift_detected OR feedback_triggered is True.
    If neither is True, logs that the system is healthy and exits.

    Injection steps:
        1. Group feedback rows by condition (ckd / thyroid)
        2. Parse features_json from each row
        3. Append to the relevant raw CSV with the corrected label
        4. Mark all injected feedback rows as used_in_training = 1

    DVC step:
        5. Run `dvc repro` from /opt/airflow
        6. DVC detects which CSVs changed (via MD5 hash)
        7. Re-runs only the affected pipeline stages automatically
        8. train_ckd.py / train_thyroid.py register new model versions

    Args passed via XCom from check_drift and check_feedback.
    """
    ti                 = ctx["ti"]
    drift_detected     = ti.xcom_pull(task_ids="check_drift",    key="drift_detected")
    feedback_triggered = ti.xcom_pull(task_ids="check_feedback", key="feedback_triggered")
    feedback_data      = ti.xcom_pull(task_ids="check_feedback", key="feedback_data") or []

    if not drift_detected and not feedback_triggered:
        log.info(
            "No retraining needed — "
            "drift=%s feedback=%s. System healthy.",
            drift_detected, feedback_triggered,
        )
        return

    log.info(
        "Retraining triggered | drift=%s feedback=%s rows=%d",
        drift_detected, feedback_triggered, len(feedback_data),
    )

    injected_ckd     = 0
    injected_thyroid = 0

    if feedback_data and os.path.exists(CKD_RAW_PATH):
        # Separate rows by condition
        ckd_rows     = [(r[1], r[2]) for r in feedback_data if r[0] == "ckd"]
        thyroid_rows = [(r[1], r[2]) for r in feedback_data if r[0] == "thyroid"]

        # Inject CKD rows
        if ckd_rows:
            try:
                ckd_df   = pd.read_csv(CKD_RAW_PATH)
                new_rows = []
                for correct_label, features_json in ckd_rows:
                    try:
                        feat_dict         = json.loads(features_json)
                        # Remove engineered features — raw CSV should not have them
                        feat_dict.pop("kidney_stress",  None)
                        feat_dict.pop("tsh_t3_ratio",   None)
                        feat_dict["class"] = correct_label
                        new_rows.append(feat_dict)
                    except Exception as e:
                        log.warning("Skipping malformed CKD row: %s", e)

                if new_rows:
                    appended = pd.concat(
                        [ckd_df, pd.DataFrame(new_rows)],
                        ignore_index=True,
                    )
                    appended.to_csv(CKD_RAW_PATH, index=False)
                    injected_ckd = len(new_rows)
                    log.info("Injected %d rows into ckd.csv", injected_ckd)
            except Exception as e:
                log.error("Failed to inject CKD rows: %s", e)
                raise

        # Inject thyroid rows
        if thyroid_rows and os.path.exists(THY_RAW_PATH):
            try:
                thy_df   = pd.read_csv(THY_RAW_PATH)
                new_rows = []
                for correct_label, features_json in thyroid_rows:
                    try:
                        feat_dict           = json.loads(features_json)
                        feat_dict.pop("tsh_t3_ratio",  None)
                        feat_dict.pop("kidney_stress", None)
                        feat_dict["target"] = correct_label
                        new_rows.append(feat_dict)
                    except Exception as e:
                        log.warning("Skipping malformed thyroid row: %s", e)

                if new_rows:
                    appended = pd.concat(
                        [thy_df, pd.DataFrame(new_rows)],
                        ignore_index=True,
                    )
                    appended.to_csv(THY_RAW_PATH, index=False)
                    injected_thyroid = len(new_rows)
                    log.info("Injected %d rows into thyroid.csv", injected_thyroid)
            except Exception as e:
                log.error("Failed to inject thyroid rows: %s", e)
                raise


    # Run dvc repro — DVC detects CSV hash changes and re-runs affected stages
    # This triggers: validate → preprocess → engineer → baselines → train
    log.info("Running dvc repro from %s", WORKDIR)
    result = subprocess.run(
        ["dvc", "repro"],
        cwd=WORKDIR,
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )},
    )

    if result.stdout:
        log.info("dvc repro stdout:\n%s", result.stdout)
    if result.stderr:
        log.warning("dvc repro stderr:\n%s", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"dvc repro failed:\n{result.stderr}")
    
    # Mark feedback used ONLY after successful dvc repro
    if injected_ckd + injected_thyroid > 0:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE feedback SET used_in_training = 1 "
            "WHERE used_in_training = 0"
        )
        conn.commit()
        conn.close()
        log.info(
            "Marked feedback rows used | ckd=%d thyroid=%d",
            injected_ckd, injected_thyroid,
        )

    log.info(
        "dvc repro complete | injected ckd=%d thyroid=%d",
        injected_ckd, injected_thyroid,
    )

    log.info(
        "dvc repro complete | injected ckd=%d thyroid=%d",
        injected_ckd, injected_thyroid,
    )


def notify(**ctx):
    """
    Send a retraining completion email via Airflow's configured SMTP.
    Falls back to log-only if email sending fails.
    """
    from airflow.utils.email import send_email

    ti                 = ctx["ti"]
    drift_detected     = ti.xcom_pull(task_ids="check_drift",    key="drift_detected")
    feedback_triggered = ti.xcom_pull(task_ids="check_feedback", key="feedback_triggered")

    if not drift_detected and not feedback_triggered:
        log.info(
            "Nightly check complete — no retraining needed. "
            "System is healthy."
        )
        return

    subject = "[VitaDoc] Retraining pipeline completed"
    reasons = []
    if drift_detected:
        reasons.append("feature drift detected")
    if feedback_triggered:
        reasons.append("user feedback threshold reached")

    html = f"""
    <h2>VitaDoc Retraining Complete</h2>
    <p>The nightly retraining pipeline finished successfully.</p>
    <ul>
      <li><strong>Triggered by:</strong> {', '.join(reasons)}</li>
      <li><strong>Drift detected:</strong> {drift_detected}</li>
      <li><strong>Feedback triggered:</strong> {feedback_triggered}</li>
      <li><strong>Run ID:</strong> {ctx['run_id']}</li>
      <li><strong>Execution date:</strong> {ctx['execution_date']}</li>
    </ul>
    <p>Check MLflow at <a href="http://localhost:5000">http://localhost:5000</a> for new model versions.</p>
    """

    to = "your-email@example.com"   # <-- replace this

    try:
        send_email(to=to, subject=subject, html_content=html)
        log.info("Retraining notification email sent to %s", to)
    except Exception as e:
        log.warning("Failed to send notification email: %s", e)
        log.info(
            "Retraining pipeline finished | drift=%s feedback=%s",
            drift_detected, feedback_triggered,
        )



with DAG(
    dag_id="vitadoc_retraining",
    default_args=DEFAULT_ARGS,
    description="Nightly retraining — drift detection and feedback injection",
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vitadoc", "retraining"],
) as dag:

    # check_drift and check_feedback run in parallel — no dependency between them
    t_check_drift    = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    t_check_feedback = PythonOperator(
        task_id="check_feedback",
        python_callable=check_feedback,
    )

    # inject_and_retrain waits for both checks
    t_inject_retrain = PythonOperator(
        task_id="inject_and_retrain",
        python_callable=inject_and_retrain,
    )

    t_notify = PythonOperator(
        task_id="notify",
        python_callable=notify,
    )

    [t_check_drift, t_check_feedback] >> t_inject_retrain >> t_notify