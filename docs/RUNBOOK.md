# Production Operations Runbook

## Telco Customer Churn Platform

---

## 1. Routine Operational Procedures

### 1.1 Verifying Service Health

Run health probe on API:

```bash
curl -i http://localhost:8000/health
```

Expected response: HTTP 200 with `status: "healthy"` and `model_loaded: true`.

### 1.2 Retraining the Production Model

Execute the automated training pipeline:

```bash
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/generate_reports.py
```

This updates `models/production_pipeline.joblib`, `models/metadata.json`, and benchmark records.

### 1.3 Rolling Back Model Artifacts

If an anomaly is detected with a newly deployed model:

1. Locate the prior version in `models/pipeline_v1.0.0.joblib`.
2. Copy over `models/production_pipeline.joblib`:

   ```bash
   cp models/pipeline_v1.0.0.joblib models/production_pipeline.joblib
   ```

3. Restart API workers:

   ```bash
   docker compose restart api
   ```

---

## 2. Incident Response Playbooks

### Playbook A: Healthcheck Failure / Container Crash

- **Symptom:** `/health` returns 503 or container restarts repeatedly.
- **Root Cause Check:** Model artifact missing or corrupted in `models/`.
- **Resolution:**
  1. Inspect container logs: `docker logs telco_churn_api`.
  2. If missing artifact, run `python scripts/train_model.py`.
  3. Re-launch containers: `docker compose up -d --build`.

### Playbook B: High Prediction Latency ($>500\text{ ms}$)

- **Symptom:** API requests timing out or dashboard lagging.
- **Root Cause Check:** Single customer requests should take $<50\text{ ms}$. If slow, check SHAP background data size or excessive batch sizes.
- **Resolution:**
  1. Ensure batch prediction endpoint is used for multi-record scoring.
  2. Scale worker processes: set `api.workers=4` in `configs/production.yaml`.
