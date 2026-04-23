#!/bin/sh
# Inject runtime environment variables into the HTML before nginx starts.
# This lets Docker Compose pass BACKEND_URL, AIRFLOW_URL etc at container start.

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
AIRFLOW_URL="${AIRFLOW_URL:-http://localhost:8080}"
MLFLOW_URL="${MLFLOW_URL:-http://localhost:5000}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3001}"

# Write a small config JS that sets window.* variables read by the app
cat > /usr/share/nginx/html/config.js <<EOF
window.BACKEND_URL = "${BACKEND_URL}";
window.AIRFLOW_URL = "${AIRFLOW_URL}";
window.MLFLOW_URL  = "${MLFLOW_URL}";
window.GRAFANA_URL = "${GRAFANA_URL}";
EOF

echo "VitaDoc frontend config injected:"
echo "  BACKEND_URL = ${BACKEND_URL}"
echo "  AIRFLOW_URL = ${AIRFLOW_URL}"
echo "  MLFLOW_URL  = ${MLFLOW_URL}"
echo "  GRAFANA_URL = ${GRAFANA_URL}"

exec nginx -g "daemon off;"