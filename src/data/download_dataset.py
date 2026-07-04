import os
import urllib.request

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    dest_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'online_retail_II.xlsx')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded successfully to {dest_path}")
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
