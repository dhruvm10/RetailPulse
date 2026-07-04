import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_data(input_path: str, output_path: str, report_path: str):
    logger.info(f"Loading data from {input_path}")
    
    # Check if the file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    try:
        # Read the Excel dataset. Online Retail II has sheets, usually 'Year 2009-2010' and 'Year 2010-2011'
        # We will read both sheets and concatenate if possible, or just the first one if it's combined.
        # Often the downloaded file has multiple sheets. Let's read all and concatenate.
        excel_data = pd.read_excel(input_path, sheet_name=None)
        df = pd.concat(excel_data.values(), ignore_index=True)
        logger.info(f"Loaded {len(df)} rows from Excel file.")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return

    # Keep track of initial shape
    initial_shape = df.shape
    stats = {"initial_rows": initial_shape[0], "initial_cols": initial_shape[1]}
    
    # 1. Standardize column names
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    logger.info(f"Standardized columns: {df.columns.tolist()}")
    
    # Rename columns to standard ones if needed (customer id -> customer_id)
    df.rename(columns={'customer_id': 'customer_id', 'invoice': 'invoice', 'stockcode': 'stockcode',
                       'description': 'description', 'quantity': 'quantity', 'invoicedate': 'invoicedate',
                       'price': 'price', 'country': 'country'}, inplace=True)
    
    # 2. Remove duplicates
    df.drop_duplicates(inplace=True)
    stats['rows_after_dedup'] = len(df)
    logger.info(f"Removed duplicates. Remaining rows: {len(df)}")
    
    # 3. Handle missing values (Drop rows where Customer ID or Description is missing)
    df.dropna(subset=['customer_id', 'description'], inplace=True)
    stats['rows_after_dropna'] = len(df)
    logger.info(f"Dropped nulls in Customer ID/Description. Remaining rows: {len(df)}")
    
    # 4. Remove cancelled orders (Invoice starts with 'C')
    # Invoice can be numeric or string, convert to string safely
    df['invoice'] = df['invoice'].astype(str)
    df = df[~df['invoice'].str.startswith('C')]
    stats['rows_after_no_cancels'] = len(df)
    logger.info(f"Removed cancelled orders. Remaining rows: {len(df)}")
    
    # 5. Remove negative quantities and prices
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df.dropna(subset=['quantity', 'price'], inplace=True)
    df = df[(df['quantity'] > 0) & (df['price'] > 0)]
    stats['rows_after_positives'] = len(df)
    logger.info(f"Removed negative/zero quantities and prices. Remaining rows: {len(df)}")
    
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
Data Types:
{df.dtypes.to_string()}
"""
    
    # Save Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)
    logger.info(f"Saved cleaning report to {report_path}")
    
    # Save Processed Data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, 'data', 'raw', 'online_retail_II.xlsx')
    output_file = os.path.join(base_dir, 'data', 'processed', 'cleaned_retail_data.csv')
    report_file = os.path.join(base_dir, 'reports', 'cleaning_summary.md')
    
    clean_data(input_file, output_file, report_file)
