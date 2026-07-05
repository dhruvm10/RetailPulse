import great_expectations as ge
import pandas as pd
import os
import yaml

def validate_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    data_path = os.path.join(base_dir, config['paths']['processed_data'])
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Run make_dataset.py first.")
        return False
        
    # Load into GE dataframe
    df = ge.read_csv(data_path)
    
    # Define expectations
    df.expect_column_to_exist("customer_id")
    df.expect_column_to_exist("revenue")
    
    df.expect_column_values_to_not_be_null("customer_id")
    df.expect_column_values_to_be_of_type("revenue", "float64")
    df.expect_column_values_to_be_between("revenue", min_value=0)
    
    # Run validation
    results = df.validate()
    
    if results["success"]:
        print("Great Expectations Validation Passed!")
        return True
    else:
        print("Data Validation Failed!")
        print(results)
        return False

if __name__ == "__main__":
    validate_data()
