import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import yaml
import logging
import joblib
import mlflow
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_churn_model(config: Dict[str, Any], base_dir: str) -> None:
    input_path = os.path.join(base_dir, config['paths']['rfm_features'])
    models_dir = os.path.join(base_dir, config['paths']['models_dir'], 'churn')
    reports_dir = os.path.join(base_dir, config['paths']['reports_dir'], 'churn')
    
    # MLflow Setup
    mlflow.set_tracking_uri(config['mlops']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['mlops']['experiments']['churn'])
    
    logger.info(f"Loading RFM data for churn prediction from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    
    churn_threshold = config['features'].get('churn_threshold_days', 90)
    df['churn'] = (df['recency'] > churn_threshold).astype(int)
    
    features = ['frequency', 'monetary', 'average_order_value', 'customer_tenure_days', 'purchase_frequency_per_day']
    X = df[features]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    xgb_config = config['models']['xgboost']
    n_estimators = xgb_config.get('n_estimators', 100)
    max_depth = xgb_config.get('max_depth', 4)
    learning_rate = xgb_config.get('learning_rate', 0.1)
    random_state = xgb_config.get('random_state', 42)
    
    with mlflow.start_run(run_name="xgboost_churn"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("churn_threshold_days", churn_threshold)
        
        logger.info("Training XGBoost Classifier...")
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"Metrics - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'xgboost_churn_model.pkl')
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")
        
        logger.info("Generating SHAP plots...")
        os.makedirs(reports_dir, exist_ok=True)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        shap_path = os.path.join(reports_dir, 'shap_summary_plot.png')
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(shap_path, artifact_path="plots")
        
        df['churn_probability'] = model.predict_proba(X)[:, 1]
        output_path = os.path.join(base_dir, config['paths']['churn_probs'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info("Churn pipeline complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_path)
    train_churn_model(config, base_dir)
