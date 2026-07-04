import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_inventory(input_path: str, output_dir: str):
    logger.info(f"Loading product features for inventory optimization from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    
    # Constants
    # Assuming lead time is 7 days and service level is 95% (Z = 1.65)
    LEAD_TIME_DAYS = 7
    Z_SCORE = 1.65
    
    # Calculate daily variance from overall variance 
    # (assuming average_demand is per purchase, but we need per day. 
    # Let's approximate daily demand by total quantity / 365)
    
    # To do this accurately, we should calculate daily demand per product.
    # For now, we'll use a simplified formula assuming 'average_demand' is daily.
    
    # Actually, let's just calculate it on the fly:
    # average_demand from build_features is the mean quantity per order.
    # To get daily average demand, we approximate: total_quantity / 365
    df['daily_avg_demand'] = df['total_quantity_sold'] / 365
    df['daily_demand_std'] = np.sqrt(df['demand_variance']) # std deviation per order
    
    # Calculate Safety Stock = Z * Standard Deviation * sqrt(Lead Time)
    df['safety_stock'] = np.ceil(Z_SCORE * df['daily_demand_std'] * np.sqrt(LEAD_TIME_DAYS))
    
    # Calculate Lead Time Demand
    df['lead_time_demand'] = np.ceil(df['daily_avg_demand'] * LEAD_TIME_DAYS)
    
    # Calculate Reorder Point
    df['reorder_point'] = df['lead_time_demand'] + df['safety_stock']
    
    # Calculate Reorder Quantity (Economic Order Quantity simplified)
    # Let's just recommend reordering 30 days worth of demand
    df['reorder_quantity'] = np.ceil(df['daily_avg_demand'] * 30)
    
    # Handle NaNs
    df.fillna(0, inplace=True)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'inventory_recommendations.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved inventory optimization recommendations to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, 'data', 'processed', 'product_features.csv')
    output_directory = os.path.join(base_dir, 'reports', 'inventory')
    optimize_inventory(input_file, output_directory)
