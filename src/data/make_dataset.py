import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.
    
    Args:
        config_path (str): The absolute or relative path to the YAML config file.
        
    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def clean_data(input_path: str, output_path: str, report_path: str, etl_config: Dict[str, Any]) -> None:
    """
    Reads raw Excel dataset, performs data cleaning/ETL based on configuration, and saves the processed CSV.
    
    Args:
        input_path (str): Path to the raw Excel dataset.
        output_path (str): Path where the cleaned CSV dataset will be saved.
        report_path (str): Path where the cleaning summary markdown report will be saved.
        etl_config (Dict[str, Any]): ETL configuration rules from config.yaml.
        
    Returns:
        None
    """
    logger.info(f"Loading data from {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist. Please ensure data is downloaded.")
        return
        
    try:
        excel_data = pd.read_excel(input_path, sheet_name=None)
        df = pd.concat(excel_data.values(), ignore_index=True)
        logger.info(f"Loaded {len(df)} rows from Excel file.")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return

    # Track row counts for the report
    initial_shape = df.shape
    stats = {"initial_rows": initial_shape[0], "initial_cols": initial_shape[1]}
    
    # 1. Standardize column names
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    
    # Rename specifically if required
    df.rename(columns={'customer_id': 'customer_id', 'invoice': 'invoice', 'stockcode': 'stockcode',
                       'description': 'description', 'quantity': 'quantity', 'invoicedate': 'invoicedate',
                       'price': 'price', 'country': 'country'}, inplace=True)
    
    # 2. Remove duplicates
    df.drop_duplicates(inplace=True)
    stats['rows_after_dedup'] = len(df)
    
    # 3. Handle missing values
    if etl_config.get('drop_null_customers', True):
        df.dropna(subset=['customer_id', 'description'], inplace=True)
    stats['rows_after_dropna'] = len(df)
    
    # 4. Remove cancelled orders
    df['invoice'] = df['invoice'].astype(str)
    if etl_config.get('remove_cancelled_orders', True):
        df = df[~df['invoice'].str.startswith('C')]
    stats['rows_after_no_cancels'] = len(df)
    
    # 5. Remove negative quantities and prices
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['quantity', 'price'], inplace=True)
    
    if etl_config.get('remove_negative_quantities', True):
        df = df[df['quantity'] > 0]
    if etl_config.get('remove_negative_prices', True):
        df = df[df['price'] > 0]
        
    stats['rows_after_positives'] = len(df)
    
    # 6. Convert Datatypes
    df['customer_id'] = df['customer_id'].astype(int).astype(str)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df.dropna(subset=['invoicedate'], inplace=True)
    stats['rows_after_date_fix'] = len(df)
    
    # 7. Generate Revenue column
    df['revenue'] = df['quantity'] * df['price']
    
    logger.info("Data Cleaning Complete.")
    
    # Generate cleaning summary report
    report_content = f"""# Data Cleaning Summary Report

- **Initial Rows:** {stats['initial_rows']}
- **Rows after Deduplication:** {stats['rows_after_dedup']}
- **Rows after Dropping Null Customer/Description:** {stats['rows_after_dropna']}
- **Rows after Removing Cancellations:** {stats['rows_after_no_cancels']}
- **Rows after Removing Negative Quantity/Price:** {stats['rows_after_positives']}
- **Final Valid Rows:** {stats['rows_after_date_fix']}
- **Total Rows Removed:** {stats['initial_rows'] - stats['rows_after_date_fix']}

## Dataset Preview
Columns: {df.columns.tolist()}
"""
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)
        
    # Save Processed Data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    # Load configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_path)
    
    # Resolve Paths
    input_file = os.path.join(base_dir, config['paths']['raw_data'])
    output_file = os.path.join(base_dir, config['paths']['processed_data'])
    report_file = os.path.join(base_dir, config['paths']['reports_dir'], 'cleaning_summary.md')
    
    # Run ETL
    clean_data(input_file, output_file, report_file, config['etl'])
