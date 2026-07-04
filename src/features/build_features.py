import pandas as pd
import numpy as np
import os
import logging
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds Recency, Frequency, Monetary and other customer features."""
    
    # Calculate a reference date (usually 1 day after the max invoice date in dataset)
    snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)
    
    # Group by customer_id to calculate RFM
    rfm = df.groupby('customer_id').agg({
        'invoicedate': lambda x: (snapshot_date - x.max()).days,
        'invoice': 'nunique',
        'revenue': 'sum'
    }).reset_index()
    
    rfm.rename(columns={
        'invoicedate': 'recency',
        'invoice': 'frequency',
        'revenue': 'monetary'
    }, inplace=True)
    
    # Additional Customer Features
    # Average Order Value
    rfm['average_order_value'] = rfm['monetary'] / rfm['frequency']
    
    # Days Since First Purchase
    first_purchase = df.groupby('customer_id')['invoicedate'].min().reset_index()
    first_purchase.rename(columns={'invoicedate': 'first_purchase_date'}, inplace=True)
    rfm = rfm.merge(first_purchase, on='customer_id')
    rfm['customer_tenure_days'] = (snapshot_date - rfm['first_purchase_date']).dt.days
    
    # Purchase Frequency (Orders per day)
    rfm['purchase_frequency_per_day'] = rfm['frequency'] / rfm['customer_tenure_days']
    rfm['purchase_frequency_per_day'] = rfm['purchase_frequency_per_day'].replace([np.inf, -np.inf], 0)
    
    return rfm

def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds product related features."""
    product_stats = df.groupby('stockcode').agg({
        'quantity': ['sum', 'mean', 'var'],
        'revenue': 'sum',
        'customer_id': 'nunique'
    }).reset_index()
    
    product_stats.columns = ['stockcode', 'total_quantity_sold', 'average_demand', 'demand_variance', 'total_product_revenue', 'unique_customers']
    product_stats['demand_variance'] = product_stats['demand_variance'].fillna(0)
    
    return product_stats

def engineer_features(input_path: str, output_dir: str):
    logger.info(f"Loading data for feature engineering: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Building Customer RFM Features...")
    rfm_df = build_rfm_features(df)
    rfm_path = os.path.join(output_dir, 'customer_rfm_features.csv')
    rfm_df.to_csv(rfm_path, index=False)
    logger.info(f"Saved RFM features to {rfm_path}")
    
    logger.info("Building Product Features...")
    product_df = build_product_features(df)
    product_path = os.path.join(output_dir, 'product_features.csv')
    product_df.to_csv(product_path, index=False)
    logger.info(f"Saved Product features to {product_path}")
    
    logger.info("Feature engineering complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, 'data', 'processed', 'cleaned_retail_data.csv')
    output_directory = os.path.join(base_dir, 'data', 'processed')
    engineer_features(input_file, output_directory)
