# VitaDoc: Low Level Design Document

## API Specification

- **Base URL:** `http://localhost:8000`
- **Protocol:** HTTP/1.1 (HTTPS in production via nginx)
- **Format:** JSON request/response bodies, multipart/form-data for PDF upload
- **CORS:** Enabled for all origins (configurable)

---

## Endpoints

### `GET /health`

**Purpose:** Liveness probe. Returns immediately if the process is alive.
Used by Docker healthcheck and container orchestration.

**Request:** None

**Response 200:**
```json
{
  "status": "ok"
}
```

**Used by:** Docker Compose healthcheck
---

### `GET /ready`

**Purpose:** Readiness probe. Confirms both ML models are loaded and
the backend is ready to serve inference requests.

**Request:** None

**Response 200 — both models loaded:**
```json
{
  "status": "ready",
  "models_loaded": ["ckd", "thyroid"],
  "model_count": 2
}
```

**Response 200 — models still loading:**
```json
{
  "status": "loading",
  "models_loaded": ["ckd"],
  "model_count": 1
}
```

**Used by:** Frontend status indicator, Docker Compose `depends_on`

---

### `POST /analyse`

**Purpose:** Accept a blood test PDF, extract lab values via OCR,
route to classifiers, generate explanations.

**Request:** `multipart/form-data`

| Field | Type    | Required | Description                        |
|-------|---------|----------|------------------------------------|
| file  | file    | Yes      | PDF blood test report              |
| age   | integer | No       | Patient age in years               |
| sex   | string  | No       | "M" or "F"                         |

**Response 200:** `AnalysisResult` (see Data Models section)

**Response 400:**
```json
{"detail": "Only PDF files are accepted."}
```

**Response 422:**
```json
{"detail": "No lab values could be extracted from this PDF. Please try manual entry."}
```

**Response 500:**
```json
{"detail": "Analysis error: <message>"}
```

---

### `POST /analyse/manual`

**Purpose:** Accept manually entered lab values as JSON and run
the full analysis pipeline. Used when PDF OCR fails or the user
prefers direct entry.

**Request body:**
```json
{
  "features": {
    "sc":   3.2,
    "bu":   75.0,
    "hemo": 8.5,
    "TSH":  0.1
  },
  "age": 58,
  "sex": "M"
}
```

| Field    | Type              | Required | Description                    |
|----------|-------------------|----------|--------------------------------|
| features | dict[str, float]  | Yes      | Lab values by canonical name   |
| age      | integer           | No       | Patient age                    |
| sex      | string            | No       | "M" or "F"                     |

**Canonical feature names:**

*CKD features:* `sc`, `bu`, `hemo`, `bgr`, `bp`, `sod`, `pot`,
`pcv`, `wc`, `rc`, `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `ba`,
`htn`, `dm`, `cad`, `appet`, `pe`, `ane`

*Thyroid features:* `TSH`, `T3`, `TT4`, `T4U`, `FTI`,
`on_thyroxine`, `query_hypothyroid`, `query_hyperthyroid`,
`on_antithyroid_medication`, `sick`, `pregnant`,
`thyroid_surgery`, `goitre`, `tumor`

*Engineered (computed automatically, do not pass):*
`kidney_stress`, `tsh_t3_ratio`

**Response 200:** `AnalysisResult` (see Data Models section)

**Response 422:**
```json
{"detail": "No feature values provided."}
```

---

### `POST /feedback`

**Purpose:** Submit a user correction for a prior prediction.
Corrections are stored in SQLite and consumed by the nightly
Airflow retraining DAG.

**Request body:**
```json
{
  "prediction_id": "a3f9b1c2",
  "condition":     "ckd",
  "correct_label": "Not CKD"
}
```

| Field         | Type   | Required | Description                         |
|---------------|--------|----------|-------------------------------------|
| prediction_id | string | Yes      | Per-condition `prediction_id` from `predictions.<condition>.prediction_id` in AnalysisResult |
| condition     | string | Yes      | "ckd" or "thyroid"                  |
| correct_label | string | Yes      | The correct class name              |

**Valid correct_label values:**
- CKD: `"CKD Detected"`, `"Not CKD"`
- Thyroid: `"Normal"`, `"Hypothyroid"`, `"Hyperthyroid"`

**Response 200:**
```json
{
  "status":      "saved",
  "feedback_id": "b7c2d9e1"
}
```

**Response 500:**
```json
{"detail": "Could not save feedback."}
```

---

### `GET /stats`

**Purpose:** Return prediction counts and pending feedback summary
for the Pipeline monitoring page.

**Request:** None

**Response 200:**
```json
{
  "prediction_counts": {
    "ckd":     {"Not CKD": 45, "CKD Detected": 32},
    "thyroid": {"Normal": 68, "Hypothyroid": 12, "Hyperthyroid": 7}
  },
  "pending_feedback": 3,
  "models_loaded":    ["ckd", "thyroid"]
}
```

---

### `GET /pipeline/runs`

**Purpose:** Return recent Airflow run history for the Pipeline page run-log
panel.

**Request:** None

**Response 200:**
```json
[
  {
    "run": "9f8a1b2c",
    "run_id_full": "scheduled__2026-04-26T02:00:00+00:00",
    "dag": "vitadoc_retraining",
    "status": "success",
    "start": "2026-04-26 02:00",
    "dur": "—",
    "trigger": "Scheduled"
  }
]
```

---

### `GET /airflow-proxy`

**Purpose:** Proxy read-only Airflow API calls via backend to avoid exposing
credentials directly in frontend JavaScript.

**Request Query Parameters:**

| Field | Type   | Required | Description |
|-------|--------|----------|-------------|
| path  | string | Yes      | Airflow API subpath (for example, `dags`) |

**Example:**

`GET /airflow-proxy?path=dags`

**Response 200:** JSON response forwarded from Airflow API.

**Response 5xx:** Backend transport/proxy error from upstream call.

---

### `GET /metrics`

**Purpose:** Prometheus scrape endpoint. Returns all instrumented
metrics in Prometheus text format.

**Request:** None

**Response 200:** Prometheus text format (Content-Type: text/plain)

**Metrics exposed:**

| Metric | Type | Description |
|--------|------|-------------|
| `vitadoc_predictions_total` | Counter | Predictions by condition and class |
| `vitadoc_inference_seconds` | Histogram | Inference latency with buckets |
| `vitadoc_last_inference_seconds` | Gauge | Most recent inference latency |
| `vitadoc_feature_drift_stddevs` | Gauge | Rolling mean drift from baseline |
| `vitadoc_ocr_coverage` | Gauge | Fraction of expected features extracted |
| `vitadoc_requests_total` | Counter | Requests by endpoint |
| `vitadoc_feedback_total` | Counter | Feedback submissions by condition |
| `vitadoc_model_version` | Gauge | Currently loaded model version |
| `vitadoc_analysis_total` | Counter | Analyses by endpoint and outcome |

### Throughput Measurement

Throughput is computed from Prometheus counters and Airflow run logs:

- API request rate: `rate(vitadoc_requests_total[1m])`
- Analysis success rate: `rate(vitadoc_analysis_total{outcome="success"}[1m])`
- Inference latency distribution: `vitadoc_inference_seconds` histogram
- Pipeline throughput: completed DAG runs per day/week from Airflow metadata

---

## Data Models

### `AnalysisResult`

Returned by both `/analyse` and `/analyse/manual`.

```json
{
  "report_id": "a3f9b1c2",
  "extracted_features": {
    "sc":            3.2,
    "bu":            75.0,
    "hemo":          8.5,
    "kidney_stress": 0.47
  },
  "flags": {
    "sc":   "HIGH",
    "bu":   "HIGH",
    "hemo": "LOW"
  },
  "predictions": {
    "ckd": {
      "label":         1,
      "confidence":    0.9234,
      "class_name":    "CKD Detected",
      "coverage":      0.48,
      "skipped":       false,
      "prediction_id": "b7c2d9e1"
    },
    "thyroid": {
      "skipped":     true,
      "skip_reason": "insufficient_data",
      "coverage":    0.12
    }
  },
  "explanations": {
    "ckd": "The analysis flagged elevated creatinine and urea levels..."
  },
  "ocr_coverage": 0.342
}
```

| Field               | Type             | Description                               |
|---------------------|------------------|-------------------------------------------|
| report_id           | string           | 8-char unique ID for this analysis        |
| extracted_features  | dict[str, float] | All lab values including engineered       |
| flags               | dict[str, str]   | NORMAL / HIGH / LOW / UNKNOWN per feature |
| predictions         | dict[str, object]| Per-condition prediction result           |
| explanations        | dict[str, str]   | Plain-English explanation per condition   |
| ocr_coverage        | float [0-1]      | Fraction of expected features present     |

### `PredictionResult` (within predictions dict)

**When not skipped:**

| Field         | Type    | Description                              |
|---------------|---------|------------------------------------------|
| label         | integer | 0 or 1 for CKD; 0, 1, or 2 for thyroid  |
| confidence    | float   | Model probability for predicted class    |
| class_name    | string  | Human-readable class label               |
| coverage      | float   | Fraction of model features present       |
| skipped       | boolean | Always false when prediction runs        |
| prediction_id | string  | ID to reference in /feedback             |

**When skipped:**

| Field       | Type    | Description                           |
|-------------|---------|---------------------------------------|
| skipped     | boolean | true                                  |
| skip_reason | string  | "insufficient_data" or "model_not_loaded" |
| coverage    | float   | Fraction of features present          |

### Class Labels

| Condition | Label | Class Name      |
|-----------|-------|-----------------|
| CKD       | 0     | Not CKD         |
| CKD       | 1     | CKD Detected    |
| Thyroid   | 0     | Normal          |
| Thyroid   | 1     | Hypothyroid     |
| Thyroid   | 2     | Hyperthyroid    |

### Flag Values

| Flag    | Meaning                              |
|---------|--------------------------------------|
| NORMAL  | Value within clinical reference range|
| HIGH    | Value above upper reference bound    |
| LOW     | Value below lower reference bound    |
| UNKNOWN | No reference range defined           |

---

## Feature Engineering at Inference Time

Engineered features are computed in `run_analysis()` before inference.
These must exactly match the computation in `src/features/engineer.py`
to prevent train/serve skew.

### `kidney_stress` — CKD composite score [0, 1]

```
components = []
if sc   present: components += ((sc   - 0.7)  / (20.0 - 0.7))  × 0.40
if bu   present: components += ((bu   - 7.0)  / (300.0 - 7.0)) × 0.35
if hemo present: components += ((17.0 - hemo) / (17.0 - 2.0))  × 0.25

kidney_stress = clip(sum(components), 0, 1)
```

### `tsh_t3_ratio` — Thyroid axis interaction

```
tsh_t3_ratio = TSH / (T3 + 0.01)
```

Epsilon 0.01 prevents division by zero when T3 is absent or zero.

---

## Coverage Threshold

A model prediction is skipped if:

```
coverage = (features present in model's expected columns)
           / (total expected columns)

skip if coverage < 0.40
```

---

## Scalability and Batch Processing Strategy

### Current Implementation

- `/analyse` processes one PDF per request.
- Backend requests are stateless, so horizontal API scaling is possible by
  running multiple backend replicas behind a load balancer/reverse proxy.
- `/health` and `/ready` support orchestration-friendly deployments.

### Batch Processing in General (Non-UI)

- **Today (no API changes):** run multiple parallel `/analyse` requests from a
  script or service (client fan-out).
- **Planned extension:** add `/analyse/batch` with `files[]` multipart input and
  per-file result objects for partial success handling.

### Guardrails for Scale

- Cap maximum files and payload size per request.
- Use bounded worker concurrency for OCR + inference.
- Emit per-file IDs for traceability and retries.
- Track batch metrics for throughput/failures/latency in Prometheus.

### Persistence Scale Path

- Current demo profile uses SQLite.
- For higher write concurrency and larger traffic, migrate persistence to
  PostgreSQL while preserving the same API contracts.

---

## SQLite Schema

### `predictions` table

| Column          | Type    | Description                          |
|-----------------|---------|--------------------------------------|
| id              | TEXT PK | 12-char UUID prefix                  |
| timestamp       | REAL    | Unix timestamp                       |
| condition       | TEXT    | "ckd" or "thyroid"                   |
| features_json   | TEXT    | JSON-serialised feature dict         |
| predicted_label | INTEGER | Model output label                   |
| predicted_class | TEXT    | Human-readable class name            |
| confidence      | REAL    | Model confidence [0, 1]              |
| used_in_training| INTEGER | 0 = unused, 1 = injected into CSV    |

### `feedback` table

| Column          | Type    | Description                          |
|-----------------|---------|--------------------------------------|
| id              | TEXT PK | 12-char UUID prefix                  |
| prediction_id   | TEXT    | References predictions.id            |
| condition       | TEXT    | "ckd" or "thyroid"                   |
| correct_label   | TEXT    | User-provided correction             |
| timestamp       | REAL    | Unix timestamp                       |
| used_in_training| INTEGER | 0 = unused, 1 = injected into CSV    |

---

## Airflow DAG Specifications

### `vitadoc_data_pipeline`

| Parameter       | Value                |
|-----------------|----------------------|
| Schedule        | `@weekly`            |
| Pool            | `vitadoc_pool` (3 slots) |
| Retries         | 1                    |
| Retry delay     | 2 minutes            |
| Email on failure| False                |

**Task dependency graph:**
```
download (1 slot)
    → validate (1 slot)
        → preprocess_ckd (1 slot) ──┐
        → preprocess_thyroid (1 slot)┤→ engineer (2 slots)
                                     │      → baselines (no pool)
                                     │      → eda (no pool)
```

### `vitadoc_retraining`

| Parameter       | Value                |
|-----------------|----------------------|
| Schedule        | `0 2 * * *` (2am)    |
| Pool            | None                 |
| Retries         | 1                    |
| Retry delay     | 5 minutes            |
| Drift threshold | 2.0 std deviations   |
| Feedback min    | 20 corrections       |
| Feedback rate   | > 15%                |

**Task dependency graph:**
```
check_drift ──┐
              ▼
check_feedback → inject_and_retrain → notify
```

---

## Environment Variables

| Variable              | Service   | Default              | Description              |
|-----------------------|-----------|----------------------|--------------------------|
| MLFLOW_TRACKING_URI   | backend   | http://localhost:5000| MLflow server URL        |
| GROQ_API_KEY          | backend   | ""                   | Groq API key (optional)  |
| BASELINE_PATH         | backend   | data/baseline_stats.json | Drift baseline file  |
| DB_PATH               | backend   | logs/vitadoc.db      | SQLite database path     |
| BACKEND_URL           | frontend  | http://localhost:8000| Backend API URL          |
| AIRFLOW_URL           | backend, frontend | http://localhost:8080| Airflow base URL (proxy/UI) |
| MLFLOW_URL            | frontend  | http://localhost:5000| MLflow UI URL            |
| GRAFANA_URL           | frontend  | http://localhost:3001| Grafana UI URL           |
| VITADOC_DB_PATH       | airflow   | /opt/airflow/logs/vitadoc.db | Shared DB path   |
| MAILTRAP_USER         | airflow   | ""                   | SMTP credentials (optional)|
| MAILTRAP_PASSWORD     | airflow   | ""                   | SMTP credentials (optional)|
