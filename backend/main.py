"""
main.py

FastAPI inference backend for VitaDoc.

Responsibilities:
    - Load CKD and Thyroid models from MLflow registry at startup
    - Extract lab values from uploaded PDFs via pdfplumber + regex
    - Route features to whichever classifiers have sufficient coverage
    - Generate plain-English explanations via Groq API
    - Log all predictions to SQLite for the retraining feedback loop
    - Expose Prometheus metrics for drift detection and monitoring
    - Provide /health and /ready endpoints for Docker orchestration

All ML logic lives here. The frontend only calls REST endpoints.
"""

import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import pickle

import httpx

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
)
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from starlette.responses import Response

log = logging.getLogger("vitadoc.backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/backend.log"),
    ],
)

# Configuration — all from environment variables so Docker can override them

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
BASELINE_PATH   = os.getenv("BASELINE_PATH", "data/baseline_stats.json")
DB_PATH         = os.getenv("DB_PATH", "logs/vitadoc.db")
REF_RANGES_PATH = os.path.join(os.path.dirname(__file__), "reference_ranges.json")

# Models are loaded into this dict at startup
MODELS: Dict[str, Any] = {}

# Rolling windows for drift computation — last 100 requests
_windows: Dict[str, deque] = {
    "sc":  deque(maxlen=100),
    "TSH": deque(maxlen=100),
    "hemo": deque(maxlen=100),
}


# Prometheus metrics

prediction_counter = Counter(
    "vitadoc_predictions_total",
    "Total predictions by condition and class",
    ["condition", "predicted_class"],
)

inference_latency = Histogram(
    "vitadoc_inference_seconds",
    "Model inference latency in seconds",
    ["model"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

feature_drift_gauge = Gauge(
    "vitadoc_feature_drift_stddevs",
    "How many std deviations the rolling mean is from the training baseline",
    ["feature"],
)

ocr_coverage_gauge = Gauge(
    "vitadoc_ocr_coverage",
    "Fraction of expected features successfully extracted from PDF",
)

request_counter = Counter(
    "vitadoc_requests_total",
    "Total API requests by endpoint",
    ["endpoint"],
)

feedback_counter = Counter(
    "vitadoc_feedback_total",
    "Total feedback submissions",
    ["condition"],
)

model_version_gauge = Gauge(
    "vitadoc_model_version",
    "Currently loaded model version",
    ["model_name"],
)

last_inference_latency = Gauge(
    "vitadoc_last_inference_seconds",
    "Last inference latency in seconds",
    ["model"],
)

analysis_outcome_counter = Counter(
    "vitadoc_analysis_total",
    "Total analyses by endpoint and outcome",
    ["endpoint", "outcome"],  # success | failure
)

# Startup / shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models and initialise database at startup.
    Called once when the FastAPI application starts.
    """
    log.info("VitaDoc backend starting up")
    os.makedirs("logs", exist_ok=True)

    # Initialise SQLite — predictions table feeds the retraining DAG
    _init_db()

    # Load reference ranges
    try:
        with open(REF_RANGES_PATH) as fh:
            app.state.ref_ranges = json.load(fh)
        log.info("Reference ranges loaded: %d features", len(app.state.ref_ranges))
    except Exception as e:
        log.error("Could not load reference ranges: %s", e)
        app.state.ref_ranges = {}

    # Load baseline stats for drift detection
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as fh:
            app.state.baseline = json.load(fh)
        log.info("Baseline stats loaded from %s", BASELINE_PATH)
    else:
        app.state.baseline = {}
        log.warning("baseline_stats.json not found — drift detection disabled")

    # Load models from local pickle if available, else fallback to MLflow registry.
    PICKLE_PATHS = {
        "ckd":     "models/ckd_model.pkl",
        "thyroid": "models/thyroid_model.pkl",
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    for model_name, key in [("VitaDoc-CKD", "ckd"), ("VitaDoc-Thyroid", "thyroid")]:
        try:
            t0          = time.time()
            pickle_path = PICKLE_PATHS[key]

            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    MODELS[key] = pickle.load(f)
                log.info(
                    "Loaded %s from local pickle in %.2fs",
                    model_name, time.time() - t0,
                )
            else:
                # Fallback to MLflow artifact store
                MODELS[key] = mlflow.pyfunc.load_model(
                    f"models:/{model_name}/Production"
                )
                log.info(
                    "Loaded %s from MLflow registry in %.2fs",
                    model_name, time.time() - t0,
                )

            # Record model version — default to 1.0 for pickle-loaded models,
            # override with registry version if MLflow has one.
            version = 1.0
            try:
                from mlflow.tracking import MlflowClient
                client   = MlflowClient(MLFLOW_URI)
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    version = float(versions[0].version)
            except Exception as mlflow_err:
                log.warning("Could not fetch MLflow version for %s: %s", model_name, mlflow_err)
            model_version_gauge.labels(model_name=key).set(version)
        except Exception as e:
            log.error("Failed to load %s: %s", model_name, e)

    log.info("Startup complete. Models loaded: %s", list(MODELS.keys()))
    yield
    log.info("VitaDoc backend shutting down")


def _init_db():
    """
    Create SQLite tables if they don't exist.

    predictions — every inference request with features and result,
                  used by the retraining DAG to detect feedback drift
    feedback    — explicit user corrections submitted via /feedback
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               TEXT PRIMARY KEY,
            timestamp        REAL,
            condition        TEXT,
            features_json    TEXT,
            predicted_label  INTEGER,
            predicted_class  TEXT,
            confidence       REAL,
            used_in_training INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id               TEXT PRIMARY KEY,
            prediction_id    TEXT,
            condition        TEXT,
            correct_label    TEXT,
            timestamp        REAL,
            used_in_training INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    log.info("SQLite database initialised at %s", DB_PATH)


# FastAPI app

app = FastAPI(
    title="VitaDoc API",
    description="Blood test analysis — ML inference, OCR extraction, AI explanations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


# Pydantic schemas

class ManualInputRequest(BaseModel):
    """Request body for the manual feature entry endpoint."""
    features: Dict[str, float]
    age:      Optional[int]   = None
    sex:      Optional[str]   = None


class FeedbackRequest(BaseModel):
    """Request body for submitting a user correction."""
    prediction_id: str
    condition:     str
    correct_label: str


class PredictionResult(BaseModel):
    """One model's prediction for a single condition."""
    label:         int
    confidence:    float
    class_name:    str
    coverage:      float
    skipped:       bool = False
    skip_reason:   Optional[str] = None


class AnalysisResult(BaseModel):
    """Full response returned to the frontend after analysis."""
    report_id:          str
    extracted_features: Dict[str, float]
    flags:              Dict[str, str]
    predictions:        Dict[str, Any]
    explanations:       Dict[str, str]
    ocr_coverage:       float


# OCR helpers

# Maps common lab report text synonyms to canonical feature names.
# This is the single place to add new synonyms — the rest of the
# pipeline uses canonical names throughout.
SYNONYM_MAP: Dict[str, str] = {
    "serum creatinine": "sc", "s. creatinine": "sc",
    "creatinine": "sc",       "creat": "sc",
    "blood urea": "bu",       "urea": "bu",       "bun": "bu",
    "haemoglobin": "hemo",    "hemoglobin": "hemo","hb": "hemo",
    "sodium": "sod",          "na+": "sod",
    "potassium": "pot",       "k+": "pot",
    "blood glucose random": "bgr", "random blood sugar": "bgr",
    "rbs": "bgr",             "glucose": "bgr",
    "packed cell volume": "pcv", "hematocrit": "pcv",
    "white blood cell": "wc", "wbc": "wc",
    "total leukocyte count": "wc", "tlc": "wc",
    "red blood cell": "rc",   "rbc count": "rc",
    "blood pressure": "bp",
    "tsh": "TSH", "thyroid stimulating hormone": "TSH",
    "t3": "T3",   "triiodothyronine": "T3",
    "t4": "TT4",  "tt4": "TT4", "total thyroxine": "TT4",
    "t4u": "T4U",
    "free thyroxine index": "FTI", "fti": "FTI",
}


def extract_features_from_pdf(pdf_bytes: bytes) -> Dict[str, float]:
    """
    Extract lab values from a blood test PDF using pdfplumber + regex.
    """
    import io
    extracted: Dict[str, float] = {}

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception as e:
        log.error("pdfplumber failed to open PDF: %s", e)
        return {}

    lines  = full_text.lower().split("\n")
    num_re = re.compile(r"[-+]?\d+\.?\d*")

    for synonym, canonical in SYNONYM_MAP.items():
        if canonical in extracted:
            continue
        for line in lines:
            if synonym in line:
                nums = num_re.findall(line)
                if nums:
                    try:
                        extracted[canonical] = float(nums[-1])
                        break
                    except ValueError:
                        continue

    log.info(
        "OCR extracted %d features: %s",
        len(extracted), list(extracted.keys()),
    )
    return extracted


# Reference range flagger

def flag_values(
    features:   Dict[str, float],
    ref_ranges: Dict,
) -> Dict[str, str]:
    """
    Compare each extracted feature against its clinical reference range.
    """
    flags: Dict[str, str] = {}
    for name, value in features.items():
        if name not in ref_ranges:
            flags[name] = "UNKNOWN"
            continue
        ref = ref_ranges[name]
        if value < ref["low"]:
            flags[name] = "LOW"
        elif value > ref["high"]:
            flags[name] = "HIGH"
        else:
            flags[name] = "NORMAL"
    return flags


# Inference routing

CKD_FEATURES = [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv",
    "wc", "rc", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet", "pe", "ane", "kidney_stress",
]
THYROID_FEATURES = [
    "age", "sex", "on_thyroxine", "TSH", "T3", "TT4", "T4U", "FTI",
    "query_hypothyroid", "query_hyperthyroid",
    "on_antithyroid_medication", "sick", "pregnant",
    "thyroid_surgery", "goitre", "tumor", "tsh_t3_ratio",
]

CKD_CLASS_NAMES     = {0: "Not CKD",     1: "CKD Detected"}
THYROID_CLASS_NAMES = {0: "Normal",       1: "Hypothyroid", 2: "Hyperthyroid"}

COVERAGE_THRESHOLD = 0.4


def run_inference(
    model_key:     str,
    features:      Dict[str, float],
    expected_cols: List[str],
    class_names:   Dict[int, str],
) -> Dict:
    """
    Run inference for one model if sufficient feature coverage exists.
    """
    model = MODELS.get(model_key)
    if model is None:
        return {"skipped": True, "skip_reason": "model_not_loaded", "coverage": 0.0}

    present  = [c for c in expected_cols if c in features]
    coverage = len(present) / len(expected_cols)

    if coverage < COVERAGE_THRESHOLD:
        log.info(
            "%s skipped — coverage %.0f%% below threshold %.0f%%",
            model_key, coverage * 100, COVERAGE_THRESHOLD * 100,
        )
        return {
            "skipped":     True,
            "skip_reason": "insufficient_data",
            "coverage":    round(coverage, 3),
        }

    # Build a single-row DataFrame with NaN for missing features
    row = {col: features.get(col, np.nan) for col in expected_cols}
    df  = pd.DataFrame([row])

    t0 = time.time()
    try:
        raw_output = model.predict(df)
    except Exception as e:
        log.error("Inference failed for %s: %s", model_key, e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    elapsed = time.time() - t0
    inference_latency.labels(model=model_key).observe(elapsed)
    last_inference_latency.labels(model=model_key).set(elapsed)

    # mlflow.pyfunc returns DataFrame or numpy array depending on model flavour
    proba = np.array(raw_output).flatten()
    if proba.ndim > 1:
        proba = proba[0]

    if len(proba) == 1:
        pred_label = int(proba[0] > 0.5)
        confidence = float(proba[0]) if pred_label == 1 else float(1 - proba[0])
    elif len(proba) == 2:
        pred_label = int(np.argmax(proba))
        confidence = float(proba[pred_label])
    else:
        pred_label = int(np.argmax(proba))
        confidence = float(proba[pred_label])
        confidence = min(confidence, 1.0)

    prediction_counter.labels(
        condition=model_key,
        predicted_class=str(pred_label),
    ).inc()

    log.info(
        "%s inference | label=%d class=%s conf=%.3f coverage=%.0f%% latency=%.3fs",
        model_key, pred_label, class_names.get(pred_label),
        confidence, coverage * 100, elapsed,
    )

    return {
        "label":       pred_label,
        "confidence":  round(confidence, 4),
        "class_name":  class_names.get(pred_label, str(pred_label)),
        "coverage":    round(coverage, 3),
        "skipped":     False,
    }


# Groq explanation agent

def get_explanation(
    condition:  str,
    features:   Dict[str, float],
    flags:      Dict[str, str],
    prediction: Dict,
) -> str:
    """
    Generate a plain-English explanation of the prediction via Groq API.
    Falls back to a templated explanation if GROQ_API_KEY is not set.
    """
    label    = prediction.get("class_name", "Unknown")
    abnormal = [k for k, v in flags.items() if v in ("HIGH", "LOW")]

    if not GROQ_API_KEY:
        abnormal_desc = ", ".join(
            f"{k} ({flags[k].lower()})" for k in abnormal[:4]
        ) or "none"
        return (
            f"The automated analysis classified your result as: {label}. "
            f"Values outside the normal range: {abnormal_desc}.\n\n"
            "Please consult a qualified doctor to interpret these results. "
            "This tool is for informational purposes only."
        )

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        system_prompt = (
            "You are a health literacy assistant. Given blood test values and their "
            "automated analysis, explain in 2 short paragraphs what the results might "
            "suggest for a non-medical person. Do not diagnose. Highlight which values "
            "are driving the classification. Always recommend seeing a doctor."
        )
        payload = {
            "condition":  condition,
            "prediction": label,
            "confidence": prediction.get("confidence"),
            "abnormal_values": {k: features.get(k) for k in abnormal},
            "flags":      flags,
        }

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": json.dumps(payload)},
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content

    except Exception as e:
        log.error("Groq API call failed: %s", e)
        return f"Result: {label}. Please consult a doctor for interpretation."


# Drift monitoring

def update_drift(features: Dict[str, float], baseline: Dict):
    """
    Update Prometheus drift gauges based on rolling mean vs training baseline.

    Computes how many standard deviations the rolling mean of each
    monitored feature has drifted from the training distribution.
    A value > 2.0 triggers the Grafana alert.

    NOTE: The window requires >= 10 samples before emitting drift values.
    During early requests, drift panels will show no data — this is expected.
    """
    for feat, window in _windows.items():
        if feat in features:
            window.append(features[feat])

        if len(window) < 10:
            continue

        # Get the baseline stats for this feature
        bstats = None
        for dataset in baseline.values():
            if feat in dataset:
                bstats = dataset[feat]
                break

        if not bstats:
            continue

        rolling_mean   = float(np.mean(window))
        baseline_mean  = bstats.get("mean", rolling_mean)
        baseline_std   = bstats.get("std", 1.0) or 1.0
        drift_stddevs  = abs(rolling_mean - baseline_mean) / baseline_std

        feature_drift_gauge.labels(feature=feat).set(drift_stddevs)
        log.debug(
            "Drift %s: rolling_mean=%.3f baseline_mean=%.3f drift=%.2f stddevs",
            feat, rolling_mean, baseline_mean, drift_stddevs,
        )


# Persistence helpers

def save_prediction(
    condition:       str,
    features:        Dict[str, float],
    predicted_label: int,
    predicted_class: str,
    confidence:      float,
) -> str:
    """
    Save a prediction to SQLite for the retraining feedback loop.
    """
    pred_id = str(uuid.uuid4())[:12]
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO predictions
               (id, timestamp, condition, features_json,
                predicted_label, predicted_class, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id, time.time(), condition,
                json.dumps(features), predicted_label,
                predicted_class, confidence,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error("Failed to save prediction to DB: %s", e)

    return pred_id


# Core analysis pipeline

def run_analysis(features: Dict[str, float]) -> AnalysisResult:
    """
    Run the full analysis pipeline on a feature dict.
    """
    report_id = str(uuid.uuid4())[:8]
    log.info("[%s] Analysis started with %d features", report_id, len(features))

    flags = flag_values(features, app.state.ref_ranges)

    # Add engineered features at inference time so they match training
    components = []
    if "sc"   in features: components.append(((features["sc"]   - 0.7)  / (20.0 - 0.7))  * 0.40)
    if "bu"   in features: components.append(((features["bu"]   - 7.0)  / (300.0 - 7.0)) * 0.35)
    if "hemo" in features: components.append(((17.0 - features["hemo"]) / (17.0 - 2.0))  * 0.25)
    if components:
        features["kidney_stress"] = float(np.clip(sum(components), 0, 1))

    if "TSH" in features and "T3" in features:
        features["tsh_t3_ratio"] = features["TSH"] / (features["T3"] + 0.01)

    predictions:  Dict[str, Any] = {}
    explanations: Dict[str, str] = {}

    for condition, expected, class_names in [
        ("ckd",     CKD_FEATURES,     CKD_CLASS_NAMES),
        ("thyroid", THYROID_FEATURES, THYROID_CLASS_NAMES),
    ]:
        result = run_inference(condition, features, expected, class_names)
        predictions[condition] = result

        if not result.get("skipped"):
            explanations[condition] = get_explanation(
                condition, features, flags, result,
            )
            pred_id = save_prediction(
                condition,
                features,
                result["label"],
                result["class_name"],
                result["confidence"],
            )
            result["prediction_id"] = pred_id


    all_expected  = set(CKD_FEATURES + THYROID_FEATURES)
    ocr_coverage  = len(set(features.keys()) & all_expected) / len(all_expected)
    ocr_coverage_gauge.set(ocr_coverage)

    update_drift(features, getattr(app.state, "baseline", {}))

    log.info(
        "[%s] Analysis complete | conditions=%s | coverage=%.2f",
        report_id,
        [k for k, v in predictions.items() if not v.get("skipped")],
        ocr_coverage,
    )

    return AnalysisResult(
        report_id=report_id,
        extracted_features=features,
        flags=flags,
        predictions=predictions,
        explanations=explanations,
        ocr_coverage=round(ocr_coverage, 3),
    )


# Endpoints

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — returns immediately if the process is alive."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe — confirms both models are loaded."""
    loaded = list(MODELS.keys())
    return {
        "status":        "ready" if len(loaded) == 2 else "loading",
        "models_loaded": loaded,
        "model_count":   len(loaded),
    }


@app.post("/analyse", response_model=AnalysisResult, tags=["inference"])
async def analyse_pdf(
    file: UploadFile       = File(...),
    age:  Optional[int]    = Form(None),
    sex:  Optional[str]    = Form(None),
):
    """
    Accept a blood test PDF, extract lab values via OCR, run classifiers.
    """
    request_counter.labels(endpoint="/analyse").inc()

    if not file.filename.lower().endswith(".pdf"):
        analysis_outcome_counter.labels(endpoint="/analyse", outcome="failure").inc()
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read error: {e}")

    features = extract_features_from_pdf(pdf_bytes)

    if age is not None:
        features["age"] = float(age)
    if sex is not None:
        features["sex"] = 1.0 if sex.upper() == "M" else 0.0

    if not features:
        raise HTTPException(
            status_code=422,
            detail=(
                "No lab values could be extracted from this PDF. "
                "Please try manual entry."
            ),
        )

    try:
        result = run_analysis(features)
        analysis_outcome_counter.labels(endpoint="/analyse", outcome="success").inc()
        return result
    except HTTPException:
        analysis_outcome_counter.labels(endpoint="/analyse", outcome="failure").inc()
        raise
    except Exception as e:
        analysis_outcome_counter.labels(endpoint="/analyse", outcome="failure").inc()
        log.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")


@app.post("/analyse/manual", response_model=AnalysisResult, tags=["inference"])
def analyse_manual(body: ManualInputRequest):
    """
    Accept manually entered lab values and run the analysis pipeline.

    BUG FIX: features must be assigned from body BEFORE the empty check.
    Previously the empty-check ran before assignment, causing a NameError
    on every manual entry request, which silently failed.
    """
    request_counter.labels(endpoint="/analyse/manual").inc()

    # FIXED: assign features first, then validate
    features = dict(body.features)

    if not features:
        analysis_outcome_counter.labels(endpoint="/analyse/manual", outcome="failure").inc()
        raise HTTPException(status_code=422, detail="No feature values provided.")

    if body.age is not None:
        features["age"] = float(body.age)
    if body.sex is not None:
        features["sex"] = 1.0 if body.sex.upper() == "M" else 0.0

    try:
        result = run_analysis(features)
        analysis_outcome_counter.labels(endpoint="/analyse/manual", outcome="success").inc()
        return result
    except HTTPException:
        analysis_outcome_counter.labels(endpoint="/analyse/manual", outcome="failure").inc()
        raise
    except Exception as e:
        analysis_outcome_counter.labels(endpoint="/analyse/manual", outcome="failure").inc()
        log.error("Manual analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")


@app.post("/feedback", tags=["feedback"])
def submit_feedback(body: FeedbackRequest):
    """
    Submit a user correction for a prior prediction.
    """
    request_counter.labels(endpoint="/feedback").inc()
    feedback_counter.labels(condition=body.condition).inc()

    feedback_id = str(uuid.uuid4())[:12]
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO feedback
               (id, prediction_id, condition, correct_label, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (feedback_id, body.prediction_id, body.condition,
             body.correct_label, time.time()),
        )
        conn.commit()
        conn.close()
        log.info(
            "Feedback saved | id=%s condition=%s correction=%s",
            feedback_id, body.condition, body.correct_label,
        )
    except Exception as e:
        log.error("Feedback DB write failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not save feedback.")

    return {"status": "saved", "feedback_id": feedback_id}


@app.get("/stats", tags=["ops"])
def get_stats():
    """
    Return summary statistics for the monitoring dashboard.
    """
    try:
        conn  = sqlite3.connect(DB_PATH)
        rows  = conn.execute(
            "SELECT condition, predicted_class, COUNT(*) as cnt "
            "FROM predictions GROUP BY condition, predicted_class"
        ).fetchall()
        fb    = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE used_in_training = 0"
        ).fetchone()[0]
        conn.close()

        counts = {}
        for condition, cls, cnt in rows:
            counts.setdefault(condition, {})[cls] = cnt

        return {
            "prediction_counts":      counts,
            "pending_feedback":       fb,
            "models_loaded":          list(MODELS.keys()),
        }
    except Exception as e:
        log.error("Stats query failed: %s", e)
        return {"error": str(e)}


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics():
    """Prometheus scrape endpoint — all custom and auto-instrumented metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/airflow-proxy", tags=["ops"])
async def airflow_proxy(path: str):
    AIRFLOW_URL  = os.getenv("AIRFLOW_URL",  "http://localhost:8080")
    AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
    AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "admin")
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{AIRFLOW_URL}/api/v1/{path}",
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=8.0,
        )
    return r.json()