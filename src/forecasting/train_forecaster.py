import pandas as pd
import os
import yaml
import logging
from prophet import Prophet
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_prophet_model(config: Dict[str, Any], base_dir: str) -> None:
    """
    Trains a Prophet model for demand forecasting and generates a 30-day prediction.
    Includes a fallback to Moving Average on Windows optimization failures.
    
    Args:
        config (Dict[str, Any]): Project configuration.
        base_dir (str): Base directory of the project.
    """
    input_path = os.path.join(base_dir, config['paths']['processed_data'])
    models_dir = os.path.join(base_dir, config['paths']['models_dir'], 'forecasting')
    reports_dir = os.path.join(base_dir, config['paths']['reports_dir'], 'forecasting')
    
    logger.info(f"Loading data for forecasting from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    
    daily_revenue = df.groupby(df['invoicedate'].dt.date)['revenue'].sum().reset_index()
    daily_revenue.columns = ['ds', 'y']
    
    try:
        logger.info(f"Training Prophet model on {len(daily_revenue)} days of data...")
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(daily_revenue)
        
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model, os.path.join(models_dir, 'prophet_model.pkl'))
        
        logger.info("Generating 30-day forecast...")
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        os.makedirs(reports_dir, exist_ok=True)
        
        fig = model.plot(forecast)
        plt.title('30-Day Revenue Forecast (Prophet)')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        fig.savefig(os.path.join(reports_dir, 'prophet_forecast.png'))
        plt.close(fig)
        
        fig_comp = model.plot_components(forecast)
        fig_comp.savefig(os.path.join(reports_dir, 'prophet_components.png'))
        plt.close(fig_comp)
        
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
        
    except Exception as e:
        logger.error(f"Prophet training failed (common on some Windows setups): {e}")
        logger.info("Falling back to Simple Moving Average Forecast for dashboard compatibility.")
        
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
    forecast_df.to_csv(os.path.join(reports_dir, '30_day_forecast.csv'), index=False)
    logger.info("Forecast generation complete.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    config = load_config(config_path)
    train_prophet_model(config, base_dir)
