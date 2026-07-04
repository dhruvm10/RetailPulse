import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_churn_model(input_path: str, models_dir: str, reports_dir: str):
    logger.info(f"Loading RFM data for churn prediction from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    
    # Define churn: If a customer hasn't purchased in the last 90 days, they are considered churned.
    # In a real setup, we'd use a temporal split to avoid leakage.
    churn_threshold = 90
    df['churn'] = (df['recency'] > churn_threshold).astype(int)
    
    # Features (excluding recency to avoid direct leakage, and customer_id)
    features = ['frequency', 'monetary', 'average_order_value', 'customer_tenure_days', 'purchase_frequency_per_day']
    X = df[features]
    y = df['churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    logger.info(f"Model Metrics - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'xgboost_churn_model.pkl'))
    
    # Generate SHAP Explainability
    logger.info("Generating SHAP explainability plots...")
    os.makedirs(reports_dir, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(reports_dir, 'shap_summary_plot.png'), bbox_inches='tight')
    plt.close()
    
    # Save test metrics report
    report_content = f"""# Churn Prediction Model Evaluation
    
- **Accuracy:** {acc:.4f}
- **Precision:** {prec:.4f}
- **Recall:** {rec:.4f}
- **F1 Score:** {f1:.4f}
- **ROC AUC:** {auc:.4f}

SHAP summary plot generated and saved.
"""
    with open(os.path.join(reports_dir, 'churn_model_evaluation.md'), "w") as f:
        f.write(report_content)
        
    # Output Churn Probabilities
    df['churn_probability'] = model.predict_proba(X)[:, 1]
    df.to_csv(os.path.join(os.path.dirname(input_path), 'customer_churn_probs.csv'), index=False)
    logger.info("Churn pipeline complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, 'data', 'processed', 'customer_rfm_features.csv')
    models_directory = os.path.join(base_dir, 'models', 'churn')
    reports_directory = os.path.join(base_dir, 'reports', 'churn')
    train_churn_model(input_file, models_directory, reports_directory)
