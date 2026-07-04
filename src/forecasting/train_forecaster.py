import pandas as pd
import os
import yaml
import logging
from prophet import Prophet
import matplotlib.pyplot as plt
import joblib
import mlflow
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_prophet_model(config: Dict[str, Any], base_dir: str) -> None:
    input_path = os.path.join(base_dir, config['paths']['processed_data'])
    models_dir = os.path.join(base_dir, config['paths']['models_dir'], 'forecasting')
    reports_dir = os.path.join(base_dir, config['paths']['reports_dir'], 'forecasting')
    
    # MLflow Setup
    mlflow.set_tracking_uri(config['mlops']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['mlops']['experiments']['forecasting'])
    
    logger.info(f"Loading data for forecasting from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    
    daily_revenue = df.groupby(df['invoicedate'].dt.date)['revenue'].sum().reset_index()
    daily_revenue.columns = ['ds', 'y']
    
    with mlflow.start_run(run_name="prophet_forecasting"):
        mlflow.log_param("yearly_seasonality", True)
        mlflow.log_param("weekly_seasonality", True)
        mlflow.log_param("daily_seasonality", False)
        
        try:
            logger.info(f"Training Prophet model on {len(daily_revenue)} days of data...")
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(daily_revenue)
            
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'prophet_model.pkl')
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")
            
            logger.info("Generating 30-day forecast...")
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            os.makedirs(reports_dir, exist_ok=True)
            
            fig = model.plot(forecast)
            plt.title('30-Day Revenue Forecast (Prophet)')
            plt.xlabel('Date')
            plt.ylabel('Revenue')
            forecast_plot = os.path.join(reports_dir, 'prophet_forecast.png')
            fig.savefig(forecast_plot)
            plt.close(fig)
            
            fig_comp = model.plot_components(forecast)
            comp_plot = os.path.join(reports_dir, 'prophet_components.png')
            fig_comp.savefig(comp_plot)
            plt.close(fig_comp)
            
            mlflow.log_artifact(forecast_plot, artifact_path="plots")
            mlflow.log_artifact(comp_plot, artifact_path="plots")
            mlflow.log_metric("prophet_used", 1)
            
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            logger.info("Falling back to Simple Moving Average Forecast.")
            mlflow.log_metric("prophet_used", 0)
            
            last_30_days_avg = daily_revenue['y'].tail(30).mean()
            last_date = daily_revenue['ds'].max()
            
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': [last_30_days_avg] * 30,
                'yhat_lower': [last_30_days_avg * 0.9] * 30,
                'yhat_upper': [last_30_days_avg * 1.1] * 30
            })
            
        os.makedirs(reports_dir, exist_ok=True)
        forecast_csv = os.path.join(reports_dir, '30_day_forecast.csv')
        forecast_df.to_csv(forecast_csv, index=False)
        mlflow.log_artifact(forecast_csv, artifact_path="predictions")
        logger.info("Forecast generation complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_path)
    train_prophet_model(config, base_dir)
