"""
evaluate_all.py

Reads both metrics JSON files and writes a markdown comparison table.
Final DVC stage — runs after both models are trained.

Usage:
    python src/models/evaluate_all.py
"""

import json
import logging
import os

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

os.makedirs("reports", exist_ok=True)


def main():
    with open("reports/ckd_metrics.json")     as f: ckd = json.load(f)
    with open("reports/thyroid_metrics.json") as f: thy = json.load(f)

    ckd_auc  = ckd.get("auc",      ckd.get("macro_f1", 0))
    thy_f1   = thy.get("macro_f1", 0)

    # Acceptance criteria — must meet these to pass
    CKD_AUC_THRESHOLD    = 0.95
    THYROID_F1_THRESHOLD = 0.50

    ckd_pass = ckd_auc  >= CKD_AUC_THRESHOLD
    thy_pass = thy_f1   >= THYROID_F1_THRESHOLD

    md = f"""# VitaDoc — Model Evaluation Report

## Results

| Model   | Best Run            | Accuracy | AUC / Macro-F1 | PASS |
|---------|---------------------|----------|----------------|------|
| CKD     | {ckd.get('run_name')} | {ckd.get('accuracy', 0):.4f} | {ckd_auc:.4f} | {'✅' if ckd_pass else '❌'} |
| Thyroid | {thy.get('run_name')} | {thy.get('accuracy', 0):.4f} | {thy_f1:.4f}  | {'✅' if thy_pass else '❌'} |

## Acceptance Criteria

| Model   | Metric   | Threshold | Achieved | Status |
|---------|----------|-----------|----------|--------|
| CKD     | AUC      | ≥ {CKD_AUC_THRESHOLD}    | {ckd_auc:.4f}  | {'PASS' if ckd_pass else 'FAIL'} |
| Thyroid | Macro-F1 | ≥ {THYROID_F1_THRESHOLD}    | {thy_f1:.4f}   | {'PASS' if thy_pass else 'FAIL'} |

## MLflow Run IDs (for reproducibility)

| Model   | Run ID |
|---------|--------|
| CKD     | `{ckd.get('run_id')}` |
| Thyroid | `{thy.get('run_id')}` |

Reproduce any run:
```bash
mlflow runs get --run-id <RUN_ID>
```
"""

    with open("reports/comparison.md", "w") as f:
        f.write(md)

    log.info("Comparison report → reports/comparison.md")
    print(md)

    if not ckd_pass or not thy_pass:
        raise SystemExit(
            f"Acceptance criteria not met — "
            f"CKD AUC={ckd_auc:.4f} (need {CKD_AUC_THRESHOLD}), "
            f"Thyroid F1={thy_f1:.4f} (need {THYROID_F1_THRESHOLD})"
        )


if __name__ == "__main__":
    main()