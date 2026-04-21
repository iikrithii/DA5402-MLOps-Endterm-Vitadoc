"""
vitadoc_pipeline.py

Airflow DAG for the VitaDoc data engineering pipeline.
Scheduled weekly — can also be triggered manually.

This DAG orchestrates the same steps as dvc repro but gives
you a management console, per-task logging, and scheduling.
Each task calls the existing feature scripts directly so
there is no logic duplication between Airflow and DVC.

Task order:
    download → validate → preprocess_ckd ──→ engineer → baselines → eda
                       → preprocess_thyroid ↗
"""

import logging
import sys
import time
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":           "vitadoc",
    "depends_on_past": False,
    "retries":         1,
    "retry_delay":     timedelta(minutes=2),
    "email_on_failure": False,
}


def task_download(**ctx):
    """Download CKD and thyroid raw data from UCI."""
    from src.features.download_data import main
    t0 = time.time()
    main()
    log.info("download completed in %.2fs", time.time() - t0)


def task_validate(**ctx):
    """Validate schema, required columns, and class balance."""
    from src.features.validate import validate_ckd, validate_thyroid
    t0 = time.time()
    validate_ckd()
    validate_thyroid()
    log.info("validation completed in %.2fs", time.time() - t0)


def task_preprocess_ckd(**ctx):
    """Clean and encode the CKD dataset."""
    import pandas as pd
    from src.features.preprocess import preprocess_ckd
    t0  = time.time()
    df  = preprocess_ckd()
    elapsed = time.time() - t0
    log.info(
        "preprocess_ckd done | rows=%d elapsed=%.2fs throughput=%.0f rows/s",
        len(df), elapsed, len(df) / elapsed,
    )


def task_preprocess_thyroid(**ctx):
    """Clean and encode the thyroid dataset."""
    import time as _t
    from src.features.preprocess import preprocess_thyroid
    t0  = _t.time()
    df  = preprocess_thyroid()
    elapsed = _t.time() - t0
    log.info(
        "preprocess_thyroid done | rows=%d elapsed=%.2fs throughput=%.0f rows/s",
        len(df), elapsed, len(df) / elapsed,
    )


def task_engineer(**ctx):
    """Add kidney_stress and tsh_t3_ratio engineered features."""
    from src.features.engineer import engineer_ckd, engineer_thyroid
    t0 = time.time()
    engineer_ckd()
    engineer_thyroid()
    log.info("engineering completed in %.2fs", time.time() - t0)


def task_baselines(**ctx):
    """Compute baseline statistics for drift detection."""
    from src.features.preprocess import compute_baselines
    t0 = time.time()
    compute_baselines()
    log.info("baselines computed in %.2fs", time.time() - t0)


def task_eda(**ctx):
    """Generate EDA plots and summary JSON."""
    from src.features.eda import run_eda
    t0 = time.time()
    run_eda()
    log.info("EDA completed in %.2fs", time.time() - t0)


with DAG(
    dag_id="vitadoc_data_pipeline",
    default_args=DEFAULT_ARGS,
    description="VitaDoc weekly data engineering pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vitadoc", "data-engineering"],
) as dag:

    t_download          = PythonOperator(task_id="download",          python_callable=task_download)
    t_validate          = PythonOperator(task_id="validate",          python_callable=task_validate)
    t_preprocess_ckd    = PythonOperator(task_id="preprocess_ckd",    python_callable=task_preprocess_ckd)
    t_preprocess_thyroid= PythonOperator(task_id="preprocess_thyroid",python_callable=task_preprocess_thyroid)
    t_engineer          = PythonOperator(task_id="engineer",          python_callable=task_engineer)
    t_baselines         = PythonOperator(task_id="baselines",         python_callable=task_baselines)
    t_eda               = PythonOperator(task_id="eda",               python_callable=task_eda)

    t_download >> t_validate
    t_validate >> [t_preprocess_ckd, t_preprocess_thyroid]
    [t_preprocess_ckd, t_preprocess_thyroid] >> t_engineer
    t_engineer >> [t_baselines, t_eda]