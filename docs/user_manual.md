# VitaDoc User Manual 

## 1. What This App Does

VitaDoc helps you understand blood test reports.

It can:
- Read a blood test PDF
- Highlight values that look low/high
- Give a simple explanation
- Flag possible kidney or thyroid risk patterns

It is an educational support tool, not a diagnosis.

---

## 2. Before You Start

You need:
- A blood test report in PDF format, or
- Your lab values if you want to enter them manually

Make sure your PDF is clear and readable (not blurry).

---

## 3. How to Analyze a PDF

1. Open VitaDoc in your browser.
2. On the Home page, drag and drop your PDF (or click to browse).
3. Click **Analyze**.
4. Wait a few seconds for results.
5. Read:
   - Condition cards (CKD / Thyroid)
   - Confidence and status
   - Flagged values (HIGH / LOW / NORMAL)
   - Plain-language explanation

---

## 4. Manual Entry

Use this if your PDF cannot be read properly/ You dont have a high quality PDF. 
1. Open the **Manual Entry** page.
2. Enter available lab values (you do not need every field).
3. Add age and sex if known.
4. Click **Analyze Manual Input**.
5. Review the same result cards and explanations.

---

## 5. Understanding the Result Screen

- **Detected / Normal labels:** Model prediction for each condition.
- **Confidence:** How sure the model is (higher is stronger confidence).
- **Coverage/Data Coverage:** How much required data was available.
- **Insufficient Data:** Not enough inputs for a reliable model run.
- **Flags:**
  - `HIGH`: Above normal range
  - `LOW`: Below normal range
  - `NORMAL`: In normal range

---

## 6. Giving Feedback (Important)

If you think a prediction is wrong:

1. Use the feedback option in the results section.
2. Select the correct label.
3. Submit.

Your feedback is stored and can be used in retraining.

---

## 7. Pipeline and Monitoring Page

You can open the Pipeline page to view:
- Recent pipeline runs
- Success/failure status
- Links to Airflow, MLflow, and Grafana

This is mostly for project/demo monitoring.

---

## 8. Common Problems and Quick Fixes

### Problem: "Only PDF files are accepted."
- Fix: Upload a `.pdf` file only.

### Problem: "No lab values could be extracted."
- Fix: Try a cleaner PDF, or use Manual Entry.

### Problem: Backend unavailable / cannot analyze
- Fix: Check if backend service is running.

### Problem: "Insufficient Data"
- Fix: Add more lab values (especially key markers like creatinine, urea, TSH, T3, T4).

---

## 9. Safety Note

VitaDoc does not replace a doctor.
Always discuss abnormal or concerning results with a qualified healthcare professional.

