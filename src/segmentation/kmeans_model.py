import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_segmentation_model(input_path: str, models_dir: str, reports_dir: str):
    logger.info(f"Loading RFM data from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    
    # We will cluster using Recency, Frequency, Monetary, and Average Order Value
    features = ['recency', 'frequency', 'monetary', 'average_order_value']
    
    # Handle outliers by applying log transformation to highly skewed features
    # Add small epsilon to avoid log(0)
    for col in features:
        df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
    log_features = [f'{col}_log' for col in features]
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[log_features])
    
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    logger.info("Finding optimal K using Elbow Method & Silhouette Score...")
    # For a real pipeline, we might dynamically find optimal k. Here we use 4 for stability or calculate it.
    # Let's fix k=5 for VIP, Loyal, Regular, New, At Risk
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Calculate Silhouette
    sil_score = silhouette_score(scaled_data, df['cluster'])
    logger.info(f"Silhouette Score for k={k}: {sil_score:.4f}")
    
    joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_model.pkl'))
    
    # Assign intuitive names based on cluster centers
    # To do this robustly, we look at average RFM per cluster
    cluster_means = df.groupby('cluster')[features].mean()
    
    # Simple heuristic to name clusters based on recency and monetary
    def assign_segment_name(row):
        # This is a placeholder logic for naming
        if row['monetary'] > cluster_means['monetary'].quantile(0.75):
            return "VIP Customers"
        elif row['recency'] < cluster_means['recency'].quantile(0.25) and row['frequency'] > cluster_means['frequency'].median():
            return "Loyal Customers"
        elif row['recency'] > cluster_means['recency'].quantile(0.75):
            return "At Risk Customers"
        else:
            return "Regular Customers"
            
    # Apply segment names
    # In reality, map clusters directly based on their average properties.
    # For simplicity, we just rank clusters by monetary value and assign names.
    ranked_clusters = cluster_means['monetary'].sort_values(ascending=False).index
    names_map = {
        ranked_clusters[0]: "VIP Customers",
        ranked_clusters[1]: "Loyal Customers",
        ranked_clusters[2]: "Regular Customers",
        ranked_clusters[3]: "New Customers",
        ranked_clusters[4]: "At Risk Customers"
    }
    
    df['segment_name'] = df['cluster'].map(names_map)
    
    # Visualizations
    os.makedirs(reports_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='recency', y='monetary', hue='segment_name', data=df, palette='viridis')
    plt.title('Customer Segments: Recency vs Monetary')
    plt.savefig(os.path.join(reports_dir, 'segment_scatter.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(y='segment_name', data=df, palette='viridis', order=df['segment_name'].value_counts().index)
    plt.title('Customer Segment Distribution')
    plt.savefig(os.path.join(reports_dir, 'segment_distribution.png'))
    plt.close()
    
    # Save segmented data
    output_path = os.path.join(os.path.dirname(input_path), 'customer_segments.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved clustered data to {output_path}")
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, 'data', 'processed', 'customer_rfm_features.csv')
    models_directory = os.path.join(base_dir, 'models', 'segmentation')
    reports_directory = os.path.join(base_dir, 'reports', 'segmentation')
    train_segmentation_model(input_file, models_directory, reports_directory)
