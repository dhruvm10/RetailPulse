import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def optimize_inventory(config: Dict[str, Any], base_dir: str) -> None:
    """
    Computes inventory optimization metrics including Safety Stock and Reorder Points.
    
    Args:
        config (Dict[str, Any]): Project configuration.
        base_dir (str): Base directory of the project.
    """
    input_path = os.path.join(base_dir, config['paths']['product_features'])
    output_dir = os.path.join(base_dir, config['paths']['reports_dir'], 'inventory')
    
    logger.info(f"Loading product features for inventory optimization from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    
    lead_time_days = config['inventory'].get('lead_time_days', 7)
    z_score = config['inventory'].get('service_level_z_score', 1.65)
    
    df['daily_avg_demand'] = df['total_quantity_sold'] / 365
    df['daily_demand_std'] = np.sqrt(df['demand_variance'])
    
    df['safety_stock'] = np.ceil(z_score * df['daily_demand_std'] * np.sqrt(lead_time_days))
    df['lead_time_demand'] = np.ceil(df['daily_avg_demand'] * lead_time_days)
    df['reorder_point'] = df['lead_time_demand'] + df['safety_stock']
    df['reorder_quantity'] = np.ceil(df['daily_avg_demand'] * 30)
    
    df.fillna(0, inplace=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'inventory_recommendations.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved inventory optimization recommendations to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_path)
    optimize_inventory(config, base_dir)
