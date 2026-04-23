# VitaDoc — Test Plan

## Acceptance Criteria

The software meets acceptance criteria when:

| Condition | Metric | Threshold | Status |
|---|---|---|---|
| CKD model | AUC-ROC | ≥ 0.95 | ✅ |
| Thyroid model | Macro-F1 | ≥ 0.50 | ✅ |
| Backend inference | Latency | < 2s | ✅ |
| Unit tests | Pass rate | 100% | ✅ |
| Integration tests | All services healthy | All up | ✅ |

## Test Categories

| Category | File | Tests | Type |
|---|---|---|---|
| Data Validation | test_validate.py | 20 | Unit |
| Preprocessing | test_preprocess.py | 11 | Unit |
| Feature Engineering | test_engineer.py | 20 | Unit |
| Model Pipeline | test_models.py | 12 | Unit |
| Backend Logic | test_backend.py | 18 | Unit |
| Frontend Logic | test_frontend.py | 20 | Unit + Integration |
| Stack Smoke Tests | test_smoke.py | 11 | Integration |
| **Total** | | **152** | |

## Test Cases by Module

### Validation (test_validate.py) — 20 cases
| ID | Test | Expected |
|---|---|---|
| V01 | Min row count — sufficient | PASS |
| V02 | Min row count — too few | FAIL detected |
| V03 | Min row count — exact boundary | PASS |
| V04 | Null rows — none | PASS |
| V05 | Null rows — fully null row | FAIL detected |
| V06 | Partial nulls — ignored | PASS |
| V07 | Required cols — all present | PASS |
| V08 | Required cols — missing | FAIL detected |
| V09 | Flexible cols — first alias | PASS |
| V10 | Flexible cols — second alias | PASS |
| V11 | Flexible cols — neither | FAIL detected |
| V12 | Multiple missing — all reported | FAIL detected |
| V13 | Target — valid CKD values | PASS |
| V14 | Target — missing column | FAIL detected |
| V15 | Target — unexpected labels warned | PASS |
| V16 | Target — valid thyroid values | PASS |
| V17 | Class balance — balanced | PASS |
| V18 | Class balance — severe imbalance | FAIL detected |
| V19 | Class balance — exact threshold | PASS |
| V20 | Class balance — missing target | PASS |

### Preprocessing (test_preprocess.py) — 11 cases
| ID | Test | Expected |
|---|---|---|
| P01 | wbcc renamed to wc | PASS |
| P02 | Target encoding ckd→1 notckd→0 | PASS |
| P03 | yes/no encoding | PASS |
| P04 | Question marks become NaN | PASS |
| P05 | No object columns after encoding | PASS |
| P06 | negative label → 0 | PASS |
| P07 | hypothyroid label → 1 | PASS |
| P08 | hyperthyroid label → 2 | PASS |
| P09 | Boolean t/f → 1/0 | PASS |
| P10 | Sex M/F → 1/0 | PASS |
| P11 | Unknown labels → NaN | PASS |

### Feature Engineering (test_engineer.py) — 20 cases
| ID | Test | Expected |
|---|---|---|
| E01 | kidney_stress column added | PASS |
| E02 | kidney_stress within [0,1] | PASS |
| E03 | kidney_stress higher for CKD | PASS |
| E04 | kidney_stress not correlated with sc | PASS |
| E05 | Original columns preserved | PASS |
| E06 | Row count unchanged | PASS |
| E07 | Missing sc handled | PASS |
| E08 | All components missing — not added | PASS |
| E09 | No new nulls introduced | PASS |
| E10 | DataFrame accepted directly | PASS |
| E11 | tsh_t3_ratio added | PASS |
| E12 | No division by zero when T3=0 | PASS |
| E13 | Ratio higher for hypothyroid | PASS |
| E14 | Ratio not correlated with TSH | PASS |
| E15 | Original columns preserved | PASS |
| E16 | Row count unchanged | PASS |
| E17 | Missing T3 handled | PASS |
| E18 | Missing TSH handled | PASS |
| E19 | No new nulls | PASS |
| E20 | DataFrame accepted directly | PASS |

### Model Pipeline (test_models.py) — 12 cases
| ID | Test | Expected |
|---|---|---|
| M01 | Param grid correct count | PASS |
| M02 | Param grid capped at max | PASS |
| M03 | Fixed params merged | PASS |
| M04 | All combos have search keys | PASS |
| M05 | CKD pipeline fits and predicts | PASS |
| M06 | CKD predict_proba shape | PASS |
| M07 | CKD handles missing values | PASS |
| M08 | CKD AUC above threshold | PASS |
| M09 | Thyroid fits 3 classes | PASS |
| M10 | Thyroid predict_proba 3 cols | PASS |
| M11 | Thyroid handles missing values | PASS |
| M12 | Thyroid macro-F1 above threshold | PASS |

### Backend Logic (test_backend.py) — 18 cases
| ID | Test | Expected |
|---|---|---|
| B01 | HIGH value flagged | PASS |
| B02 | LOW value flagged | PASS |
| B03 | NORMAL value | PASS |
| B04 | Unknown feature | PASS |
| B05 | Boundary at low | PASS |
| B06 | Boundary at high | PASS |
| B07 | Multiple features | PASS |
| B08 | Empty features | PASS |
| B09 | Creatinine maps to sc | PASS |
| B10 | TSH maps correctly | PASS |
| B11 | Haemoglobin maps | PASS |
| B12 | HB alias maps | PASS |
| B13 | All targets known | PASS |
| B14 | Empty input returns dict | PASS |
| B15 | Invalid PDF returns dict | PASS |
| B16 | kidney_stress high for CKD | PASS |
| B17 | kidney_stress low for normal | PASS |
| B18 | TSH/T3 ratio no div by zero | PASS |

### Frontend Logic + HTML (test_frontend.py) — 20 cases

| ID | Test | Expected |
|---|---|---|
| F01 | HIGH badge renders correct class | PASS |
| F02 | LOW badge renders correct class | PASS |
| F03 | NORMAL badge renders correct class | PASS |
| F04 | Unknown badge uses default styling | PASS |
| F05 | Confidence bar at 100% renders correctly | PASS |
| F06 | Confidence bar at 0% renders correctly | PASS |
| F07 | Confidence bar at 50% renders correctly | PASS |
| F08 | CKD Detected classified as positive | PASS |
| F09 | Hypothyroid classified as positive | PASS |
| F10 | Non-positive class correctly identified | PASS |
| F11 | CKD feature list has correct length | PASS |
| F12 | Thyroid feature list has correct length | PASS |
| F13 | CKD and Thyroid features are disjoint | PASS |
| F14 | All CKD features mapped to display names | PASS |
| F15 | All Thyroid features mapped to display names | PASS |
| F16 | Valid numeric input stored correctly | PASS |
| F17 | Empty input removes field | PASS |
| F18 | Invalid input removed from state | PASS |
| F19 | HTML contains core structure (doctype, sidebar, main) | PASS |
| F20 | HTML contains key UI elements and endpoints | PASS |

### Smoke Tests (test_smoke.py) — 11 cases
| ID | Test | Expected |
|---|---|---|
| S01 | Backend /health | 200 ok |
| S02 | Backend /ready — 2 models | model_count=2 |
| S03 | MLflow healthy | 200 |
| S04 | Prometheus healthy | 200 |
| S05 | Grafana healthy | database=ok |
| S06 | Prometheus scraping backend | health=up |
| S07 | Manual analysis valid structure | All fields present |
| S08 | Prediction saved to DB | stats returns data |
| S09 | Feedback saved | status=saved |
| S10 | Prometheus metrics populated | vitadoc_requests_total present |
| S11 | Inference latency < 2s | elapsed < 2.0s |

## Running Tests

```bash
# Unit tests only (no services required)
pytest tests/ -v -m "not smoke and not integration" \
  --html=reports/test_report.html --self-contained-html

# All tests including integration (requires docker compose up -d)
pytest tests/ -v \
  --html=reports/test_report.html --self-contained-html

# Smoke tests only
pytest tests/test_smoke.py -v -m smoke
```

## Test Report
Generated at: `reports/test_report.html`