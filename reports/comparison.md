# VitaDoc — Model Evaluation Report

## Results

| Model   | Best Run            | Accuracy | AUC / Macro-F1 | PASS |
|---------|---------------------|----------|----------------|------|
| CKD     | ckd_xgb_ne100_d3_lr0.05 | 1.0000 | 1.0000 | ✅ |
| Thyroid | thy_rf_smote_ne100_d5 | 0.8889 | 0.6980  | ✅ |

## Acceptance Criteria

| Model   | Metric   | Threshold | Achieved | Status |
|---------|----------|-----------|----------|--------|
| CKD     | AUC      | ≥ 0.95    | 1.0000  | PASS |
| Thyroid | Macro-F1 | ≥ 0.5    | 0.6980   | PASS |

## MLflow Run IDs (for reproducibility)

| Model   | Run ID |
|---------|--------|
| CKD     | `e752cad7210a4ea69aab0d1ee71651df` |
| Thyroid | `fa060684fbd54bb3a052613594a2ae54` |

Reproduce any run:
```bash
mlflow runs get --run-id <RUN_ID>
```
