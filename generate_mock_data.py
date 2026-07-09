import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_mock_data(num_rows=5000):
    np.random.seed(42)
    random.seed(42)
    
    # Categories and pricing
    categories = ['Clothing', 'Electronics', 'Beauty', 'Home', 'Sports']
    base_prices = {'Clothing': 50, 'Electronics': 300, 'Beauty': 30, 'Home': 100, 'Sports': 80}
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_rows):
        t_id = i + 1
        date = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(8, 20))
        c_id = f"CUST{random.randint(1, 1000):04d}"
        gender = random.choice(['Male', 'Female'])
        age = random.randint(18, 70)
        
        category = random.choice(categories)
        qty = random.randint(1, 5)
        
        # Add some variance to price
        price = round(base_prices[category] * random.uniform(0.8, 1.2), 2)
        total = round(qty * price, 2)
        
        data.append([t_id, date.strftime('%Y-%m-%d'), c_id, gender, age, category, qty, price, total])
        
    df = pd.DataFrame(data, columns=[
        'Transaction_ID', 'Date', 'Customer_ID', 'Gender', 'Age', 
        'Product_Category', 'Quantity', 'Price', 'Total_Amount'
    ])
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/retail_sales_dataset.csv', index=False)
    print(f"Generated {num_rows} rows of mock data at data/raw/retail_sales_dataset.csv")

if __name__ == "__main__":
    generate_mock_data()
