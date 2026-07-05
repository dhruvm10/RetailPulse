from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter, Gauge
import uvicorn
import yaml
import os

app = FastAPI()

# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Define custom metrics
MODEL_ACCURACY = Gauge('retailpulse_model_mape', 'MAPE of the forecasting model')
DATA_DRIFT_SCORE = Gauge('retailpulse_drift_score', 'Data drift detected in recent batch')
PIPELINE_RUNS = Counter('retailpulse_pipeline_runs_total', 'Total number of Airflow pipeline runs')

@app.get("/")
def read_root():
    return {"status": "Metrics server is running on /metrics"}

def start_metrics_server():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    port = config.get('observability', {}).get('prometheus_port', 8000)
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_metrics_server()
