"""
vitadoc_pipeline.py

Airflow DAG for the VitaDoc weekly data engineering pipeline.

Pool: vitadoc_pool (3 slots)
    All resource-intensive tasks draw from this pool.
    With 3 slots, preprocess_ckd and preprocess_thyroid run
    in parallel (1 slot each) after validate (1 slot) finishes.
    Engineer waits for both, then runs with 2 slots (signals it
    is heavier — processes both datasets in one call).

Dynamic mapping: TASK_SLOTS maps task_id → slot count.
    Adding a new task only requires one line in TASK_SLOTS.
    The make_task() factory reads from it automatically.

Baselines and EDA do not use the pool — they are read-only
aggregations that impose no meaningful CPU load.
"""

import logging
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

log     = logging.getLogger(__name__)
PYTHON  = sys.executable
WORKDIR = "/opt/airflow"

DEFAULT_ARGS = {
    "owner":            "vitadoc",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
}

POOL_NAME = "vitadoc_pool"

# Dynamic slot mapping — single source of truth.
# Tasks not listed here run without a pool (lightweight tasks).
# preprocess_ckd + preprocess_thyroid = 2 slots total → run in parallel.
# engineer = 2 slots → signals it is heavier, blocks both preprocess slots.
TASK_SLOTS = {
    "download":           1,
    "validate":           1,
    "preprocess_ckd":     1,
    "preprocess_thyroid": 1,
    "engineer":           2,
}


def _run(script: str, args: list = None):
    """
    Run a Python script as a subprocess and raise on failure.

    Using subprocess isolates each task's Python environment
    and gives clean per-task logs in the Airflow UI.
    """
    cmd    = [PYTHON, script] + (args or [])
    result = subprocess.run(
        cmd, cwd=WORKDIR, capture_output=True, text=True
    )
    if result.stdout:
        log.info(result.stdout)
    if result.stderr:
        log.warning(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{result.stderr}")


def task_download(**ctx):
    _run("src/features/download_data.py")

def task_validate(**ctx):
    _run("src/features/validate.py", ["both"])

def task_preprocess_ckd(**ctx):
    _run("src/features/preprocess.py", ["ckd"])

def task_preprocess_thyroid(**ctx):
    _run("src/features/preprocess.py", ["thyroid"])

def task_engineer(**ctx):
    _run("src/features/engineer.py", ["both"])

def task_baselines(**ctx):
    _run("src/features/preprocess.py", ["baselines"])

def task_eda(**ctx):
    _run("src/features/eda.py")


def make_task(dag, task_id, callable_fn):
    """
    Create a PythonOperator, pulling pool slot count from TASK_SLOTS.

    If the task_id is not in TASK_SLOTS it runs without a pool.
    This means adding a new pooled task only requires one entry
    in TASK_SLOTS above — nothing else changes.
    """
    slots = TASK_SLOTS.get(task_id)
    kwargs = dict(
        task_id=task_id,
        python_callable=callable_fn,
        dag=dag,
    )
    if slots is not None:
        kwargs["pool"]       = POOL_NAME
        kwargs["pool_slots"] = slots

    return PythonOperator(**kwargs)


with DAG(
    dag_id="vitadoc_data_pipeline",
    default_args=DEFAULT_ARGS,
    description="VitaDoc weekly data engineering pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["vitadoc", "data-engineering"],
) as dag:

    t_download           = make_task(dag, "download",           task_download)
    t_validate           = make_task(dag, "validate",           task_validate)
    t_preprocess_ckd     = make_task(dag, "preprocess_ckd",     task_preprocess_ckd)
    t_preprocess_thyroid = make_task(dag, "preprocess_thyroid", task_preprocess_thyroid)
    t_engineer           = make_task(dag, "engineer",           task_engineer)
    t_baselines          = make_task(dag, "baselines",          task_baselines)
    t_eda                = make_task(dag, "eda",                task_eda)

    t_download >> t_validate
    t_validate >> [t_preprocess_ckd, t_preprocess_thyroid]
    [t_preprocess_ckd, t_preprocess_thyroid] >> t_engineer
    t_engineer >> [t_baselines, t_eda]