import pandas as pd
import pytest
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.build_features import build_rfm_features

def test_rfm_features():
    # Create sample data
    data = {
        'customer_id': ['1', '1', '2'],
        'invoicedate': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10']),
        'invoice': ['INV1', 'INV2', 'INV3'],
        'revenue': [100.0, 150.0, 200.0]
    }
    df = pd.DataFrame(data)
    
    rfm = build_rfm_features(df)
    
    assert len(rfm) == 2
    
    # Check customer 1
    c1 = rfm[rfm['customer_id'] == '1'].iloc[0]
    assert c1['frequency'] == 2
    assert c1['monetary'] == 250.0
    assert c1['average_order_value'] == 125.0
    
    # Check customer 2
    c2 = rfm[rfm['customer_id'] == '2'].iloc[0]
    assert c2['frequency'] == 1
    assert c2['monetary'] == 200.0
