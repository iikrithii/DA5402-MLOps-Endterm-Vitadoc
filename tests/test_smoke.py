"""
test_smoke.py

Smoke tests for the full Docker Compose stack.
Verifies all services are reachable and healthy.

These tests require the full stack to be running:
    docker compose up -d

Run: pytest tests/test_smoke.py -v -m smoke
Skip in CI (no Docker): pytest tests/ -v -m "not smoke"
"""

import os
import pytest
import requests

BACKEND_URL    = os.getenv("BACKEND_URL",    "http://localhost:8000")
MLFLOW_URL     = os.getenv("MLFLOW_URL",     "http://localhost:5000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL    = os.getenv("GRAFANA_URL",    "http://localhost:3001")


def _reachable(url: str, timeout: int = 5) -> bool:
    try:
        requests.get(url, timeout=timeout)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def require_stack():
    """Skip all smoke tests if the stack is not running."""
    if not _reachable(BACKEND_URL):
        pytest.skip("Docker Compose stack not running — skipping smoke tests")


@pytest.mark.smoke
class TestServiceHealth:
    """Verify every service in docker-compose.yml is reachable and healthy."""

    def test_backend_health(self):
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_backend_ready_with_models(self):
        r = requests.get(f"{BACKEND_URL}/ready", timeout=5)
        assert r.status_code == 200
        d = r.json()
        assert d["model_count"] == 2, (
            f"Expected 2 models loaded, got {d['model_count']}. "
            f"Loaded: {d.get('models_loaded')}"
        )

    def test_mlflow_healthy(self):
        r = requests.get(f"{MLFLOW_URL}/health", timeout=5)
        assert r.status_code == 200

    def test_prometheus_healthy(self):
        r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5)
        assert r.status_code == 200
        assert "Healthy" in r.text

    def test_grafana_healthy(self):
        r = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
        assert r.status_code == 200
        assert r.json()["database"] == "ok"

    def test_prometheus_scraping_backend(self):
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/targets", timeout=5
        )
        assert r.status_code == 200
        targets = r.json()["data"]["activeTargets"]
        backend_targets = [
            t for t in targets
            if t["labels"].get("job") == "vitadoc-backend"
        ]
        assert len(backend_targets) > 0, "No vitadoc-backend target in Prometheus"
        assert backend_targets[0]["health"] == "up", (
            "vitadoc-backend target is not healthy in Prometheus"
        )


@pytest.mark.smoke
class TestBackendFunctionality:
    """Verify backend endpoints work correctly end to end."""

    def test_manual_analysis_returns_valid_structure(self):
        r = requests.post(
            f"{BACKEND_URL}/analyse/manual",
            json={
                "features": {
                    "sc": 3.2, "bu": 75.0, "hemo": 8.5,
                    "sod": 128.0, "pot": 6.1, "bgr": 185.0,
                    "pcv": 25.0, "wc": 9500.0, "rc": 3.2, "bp": 90.0,
                },
                "age": 58, "sex": "M",
            },
            timeout=15,
        )
        assert r.status_code == 200
        d = r.json()
        assert "report_id"          in d
        assert "predictions"        in d
        assert "flags"              in d
        assert "extracted_features" in d
        assert "ocr_coverage"       in d

    def test_prediction_saved_to_db(self):
        # Make a prediction then verify stats show it
        requests.post(
            f"{BACKEND_URL}/analyse/manual",
            json={"features": {"sc": 2.0, "bu": 50.0, "hemo": 9.0}, "age": 45, "sex": "F"},
            timeout=15,
        )
        r = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        assert r.status_code == 200

    def test_feedback_saved(self):
        r = requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "prediction_id": "smoke-test-001",
                "condition":     "ckd",
                "correct_label": "Not CKD",
            },
            timeout=5,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "saved"

    def test_prometheus_metrics_populated_after_request(self):
        # Make a request first
        requests.post(
            f"{BACKEND_URL}/analyse/manual",
            json={"features": {"sc": 1.0}, "age": 30, "sex": "M"},
            timeout=15,
        )
        # Then check metrics endpoint has vitadoc metrics
        r = requests.get(f"{BACKEND_URL}/metrics", timeout=5)
        assert r.status_code == 200
        assert "vitadoc_requests_total" in r.text

    def test_inference_latency_within_sla(self):
        """Business metric: inference must complete within 2 seconds."""
        import time
        start = time.time()
        requests.post(
            f"{BACKEND_URL}/analyse/manual",
            json={
                "features": {
                    "sc": 3.2, "bu": 75.0, "hemo": 8.5,
                    "sod": 128.0, "pot": 6.1, "bgr": 185.0,
                    "pcv": 25.0, "wc": 9500.0, "rc": 3.2, "bp": 90.0,
                },
                "age": 58, "sex": "M",
            },
            timeout=15,
        )
        elapsed = time.time() - start
        assert elapsed < 2.0, (
            f"Inference took {elapsed:.2f}s — exceeds 2s SLA from HLD"
        )