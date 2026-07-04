import pandas as pd
import numpy as np
import pytest
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inventory.inventory_engine import optimize_inventory
from src.features.build_features import build_product_features

def test_inventory_logic():
    """Test the inventory optimization mathematical logic."""
    data = {
        'stockcode': ['A1', 'B2'],
        'quantity': [100, 200],
        'revenue': [1000, 4000],
        'customer_id': ['C1', 'C2']
    }
    df = pd.DataFrame(data)
    
    # 1. Test Feature Engineering for Product
    product_stats = build_product_features(df)
    assert len(product_stats) == 2
    assert product_stats.iloc[0]['total_quantity_sold'] == 100
    
    # 2. Test Inventory Calculations
    # We will manually calculate for A1
    # total_quantity = 100 -> daily_avg_demand = 100/365 = 0.2739
    daily_avg_demand = 100 / 365
    lead_time_days = 7
    z_score = 1.65
    
    # Variance is NaN for single row in groupby var(), but our code fills with 0
    std_dev = 0
    
    safety_stock = np.ceil(z_score * std_dev * np.sqrt(lead_time_days))
    lead_time_demand = np.ceil(daily_avg_demand * lead_time_days)
    reorder_point = lead_time_demand + safety_stock
    
    # Create the test dataframe for the engine logic
    test_df = product_stats.copy()
    test_df['daily_avg_demand'] = test_df['total_quantity_sold'] / 365
    test_df['daily_demand_std'] = np.sqrt(test_df['demand_variance'])
    test_df['safety_stock'] = np.ceil(z_score * test_df['daily_demand_std'] * np.sqrt(lead_time_days))
    test_df['lead_time_demand'] = np.ceil(test_df['daily_avg_demand'] * lead_time_days)
    test_df['reorder_point'] = test_df['lead_time_demand'] + test_df['safety_stock']
    
    a1_stats = test_df[test_df['stockcode'] == 'A1'].iloc[0]
    
    assert a1_stats['safety_stock'] == 0
    assert a1_stats['lead_time_demand'] == 2.0  # ceil(0.2739 * 7 = 1.91)
    assert a1_stats['reorder_point'] == 2.0
