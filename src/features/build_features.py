import pandas as pd
import numpy as np
import os
import yaml
import logging
import warnings
from typing import Dict, Any

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds Recency, Frequency, Monetary and other customer features.
    
    Args:
        df (pd.DataFrame): Cleaned transaction data.
        
    Returns:
        pd.DataFrame: A dataframe containing customer-level features.
    """
    snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_id').agg({
        'invoicedate': lambda x: (snapshot_date - x.max()).days,
        'invoice': 'nunique',
        'revenue': 'sum'
    }).reset_index()
    
    rfm.rename(columns={'invoicedate': 'recency', 'invoice': 'frequency', 'revenue': 'monetary'}, inplace=True)
    
    rfm['average_order_value'] = rfm['monetary'] / rfm['frequency']
    
    first_purchase = df.groupby('customer_id')['invoicedate'].min().reset_index()
    first_purchase.rename(columns={'invoicedate': 'first_purchase_date'}, inplace=True)
    rfm = rfm.merge(first_purchase, on='customer_id')
    rfm['customer_tenure_days'] = (snapshot_date - rfm['first_purchase_date']).dt.days
    
    rfm['purchase_frequency_per_day'] = rfm['frequency'] / rfm['customer_tenure_days']
    rfm['purchase_frequency_per_day'] = rfm['purchase_frequency_per_day'].replace([np.inf, -np.inf], 0)
    
    return rfm

def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds product-level features for inventory optimization.
    
    Args:
        df (pd.DataFrame): Cleaned transaction data.
        
    Returns:
        pd.DataFrame: A dataframe containing product-level statistics.
    """
    product_stats = df.groupby('stockcode').agg({
        'quantity': ['sum', 'mean', 'var'],
        'revenue': 'sum',
        'customer_id': 'nunique'
    }).reset_index()
    
    product_stats.columns = ['stockcode', 'total_quantity_sold', 'average_demand', 'demand_variance', 'total_product_revenue', 'unique_customers']
    product_stats['demand_variance'] = product_stats['demand_variance'].fillna(0)
    
    return product_stats

def engineer_features(config: Dict[str, Any], base_dir: str) -> None:
    """
    Main pipeline to execute feature engineering.
    
    Args:
        config (Dict[str, Any]): Project configuration.
        base_dir (str): Base directory of the project.
    """
    input_path = os.path.join(base_dir, config['paths']['processed_data'])
    logger.info(f"Loading data for feature engineering: {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    
    logger.info("Building Customer RFM Features...")
    rfm_df = build_rfm_features(df)
    rfm_path = os.path.join(base_dir, config['paths']['rfm_features'])
    os.makedirs(os.path.dirname(rfm_path), exist_ok=True)
    rfm_df.to_csv(rfm_path, index=False)
    
    logger.info("Building Product Features...")
    product_df = build_product_features(df)
    product_path = os.path.join(base_dir, config['paths']['product_features'])
    os.makedirs(os.path.dirname(product_path), exist_ok=True)
    product_df.to_csv(product_path, index=False)
    
    logger.info("Feature engineering complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_path)
    engineer_features(config, base_dir)
