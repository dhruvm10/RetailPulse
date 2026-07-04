import pandas as pd
import os
import yaml
import logging
from typing import Dict, Any
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def generate_drift_report(config: Dict[str, Any], base_dir: str) -> None:
    """
    Generates an Evidently AI Data Drift report comparing current data to historical baseline.
    Since we don't have real-time data yet, we simulate drift by splitting the current dataset temporally.
    
    Args:
        config (Dict[str, Any]): Project configuration.
        base_dir (str): Base directory of the project.
    """
    input_path = os.path.join(base_dir, config['paths']['processed_data'])
    reports_dir = os.path.join(base_dir, config['paths']['reports_dir'], 'mlops')
    
    logger.info(f"Loading data for Drift Analysis from {input_path}")
    if not os.path.exists(input_path):
        logger.error("Input file does not exist.")
        return
        
    df = pd.read_csv(input_path)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    df = df.sort_values(by='invoicedate')
    
    # Split data to simulate reference (old) vs current (new) data
    split_index = int(len(df) * 0.7)
    reference_data = df.iloc[:split_index]
    current_data = df.iloc[split_index:]
    
    logger.info("Generating Evidently AI Data Drift Report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    
    # Selecting numerical columns to check for drift
    cols_to_check = ['quantity', 'price', 'revenue']
    drift_report.run(reference_data=reference_data[cols_to_check], current_data=current_data[cols_to_check])
    
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'data_drift_report.html')
    drift_report.save_html(report_path)
    
    logger.info(f"Drift report generated and saved to {report_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_path)
    generate_drift_report(config, base_dir)
