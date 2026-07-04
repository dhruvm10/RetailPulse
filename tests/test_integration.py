import os
import subprocess
import pytest

# Define the base directory (where run_pipeline.ps1 and src/ reside)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def run_script(script_path):
    """Utility to run a Python script as a subprocess and assert success."""
    full_path = os.path.join(BASE_DIR, script_path)
    result = subprocess.run(['python', full_path], cwd=BASE_DIR, capture_output=True, text=True)
    return result

@pytest.mark.integration
def test_full_pipeline_execution():
    """
    Tests the end-to-end execution of the RetailPulse ML pipeline.
    This ensures that data flows correctly from raw extraction down to final ML model outputs without crashing.
    """
    scripts = [
        # Note: We skip download_dataset.py in automated tests to avoid large network calls
        'src/data/make_dataset.py',
        'src/features/build_features.py',
        'src/features/eda_profiling.py',
        'src/segmentation/kmeans_model.py',
        'src/forecasting/train_forecaster.py',
        'src/churn/churn_model.py',
        'src/inventory/inventory_engine.py',
        'src/mlops/data_drift.py'
    ]
    
    for script in scripts:
        print(f"Testing integration of: {script}")
        res = run_script(script)
        # Check if the process exited cleanly (code 0)
        assert res.returncode == 0, f"Script {script} failed with error:\n{res.stderr}"

    # Verify that the critical output files were generated
    expected_outputs = [
        'data/processed/cleaned_retail_data.csv',
        'data/processed/customer_segments.csv',
        'data/processed/customer_churn_probs.csv',
        'reports/forecasting/30_day_forecast.csv',
        'reports/inventory/inventory_recommendations.csv',
        'reports/mlops/data_drift_report.html'
    ]
    
    for output in expected_outputs:
        path = os.path.join(BASE_DIR, output)
        assert os.path.exists(path), f"Expected output file not found: {path}"
