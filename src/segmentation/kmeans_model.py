import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
import logging
import joblib
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_segmentation_model(config: Dict[str, Any], base_dir: str) -> None:
    """
    Trains a K-Means clustering model for customer segmentation.
    
    Args:
        config (Dict[str, Any]): Project configuration.
        base_dir (str): Base directory of the project.
    """
    input_path = os.path.join(base_dir, config['paths']['rfm_features'])
    models_dir = os.path.join(base_dir, config['paths']['models_dir'], 'segmentation')
    reports_dir = os.path.join(base_dir, config['paths']['reports_dir'], 'segmentation')
    
    logger.info(f"Loading RFM data from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    
    features = ['recency', 'frequency', 'monetary', 'average_order_value']
    
    for col in features:
        df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
    log_features = [f'{col}_log' for col in features]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[log_features])
    
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    k = config['models']['kmeans'].get('k', 5)
    random_state = config['models']['kmeans'].get('random_state', 42)
    
    logger.info(f"Training K-Means with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    sil_score = silhouette_score(scaled_data, df['cluster'])
    logger.info(f"Silhouette Score for k={k}: {sil_score:.4f}")
    
    joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_model.pkl'))
    
    cluster_means = df.groupby('cluster')[features].mean()
    ranked_clusters = cluster_means['monetary'].sort_values(ascending=False).index
    
    names_map = {
        ranked_clusters[0]: "VIP Customers",
        ranked_clusters[1]: "Loyal Customers",
        ranked_clusters[2]: "Regular Customers",
        ranked_clusters[3]: "New Customers",
        ranked_clusters[4]: "At Risk Customers"
    }
    
    df['segment_name'] = df['cluster'].map(names_map)
    
    os.makedirs(reports_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='recency', y='monetary', hue='segment_name', data=df)
    plt.title('Customer Segments: Recency vs Monetary')
    plt.savefig(os.path.join(reports_dir, 'segment_scatter.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    # Fix for seaborn Future Warning (palette without hue)
    sns.countplot(y='segment_name', data=df, hue='segment_name', palette='viridis', order=df['segment_name'].value_counts().index, legend=False)
    plt.title('Customer Segment Distribution')
    plt.savefig(os.path.join(reports_dir, 'segment_distribution.png'))
    plt.close()
    
    output_path = os.path.join(base_dir, config['paths']['segments'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved clustered data to {output_path}")
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_path)
    train_segmentation_model(config, base_dir)
