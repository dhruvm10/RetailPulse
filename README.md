# RetailPulse - AI-Powered Retail Analytics Platform 🛒

![RetailPulse](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Docker](https://img.shields.io/badge/Docker-Supported-blue)

RetailPulse is a complete, enterprise-grade AI retail analytics platform designed to solve real-world retail business problems using data analytics, machine learning, forecasting, customer intelligence, inventory optimization, explainable AI, and MLOps best practices.

## 🚀 Features

- **Demand Forecasting:** Predicts future sales trends using Facebook's Prophet and deep learning (LSTM) models.
- **Customer Segmentation:** Groups customers into actionable segments (VIP, At-Risk, Loyal) using K-Means Clustering on RFM features.
- **Churn Prediction:** Predicts the likelihood of a customer stopping purchases using XGBoost and SHAP for explainability.
- **Inventory Optimization:** Recommends optimal safety stock, reorder points, and reorder quantities to minimize stockouts and holding costs.
- **Enterprise Dashboard:** A beautiful, interactive Streamlit application with rich Plotly visualizations to monitor all business KPIs.
- **MLOps Integrated:** MLflow for experiment tracking and Evidently AI for data/model drift monitoring.

## 📂 Architecture

```text
retailpulse/
├── data/                  # Raw and processed data
├── src/
│   ├── data/              # ETL pipeline (make_dataset.py)
│   ├── features/          # Feature engineering (build_features.py, eda_profiling.py)
│   ├── forecasting/       # Prophet/LSTM models
│   ├── segmentation/      # K-Means clustering
│   ├── churn/             # XGBoost churn model with SHAP
│   └── inventory/         # Inventory optimization engine
├── dashboard/             # Streamlit application (app.py)
├── models/                # Saved ML models (.pkl)
├── reports/               # EDA charts, SHAP plots, evaluation metrics
├── tests/                 # Pytest unit and data validation tests
├── .github/workflows/     # CI/CD pipelines
├── Dockerfile             # Container configuration
└── docker-compose.yml     # Multi-container setup (App + MLflow)
```

## ⚙️ Installation & Usage

### Method 1: Docker (Recommended)
1. Clone the repository and navigate to the project directory.
2. Run `docker-compose up --build`
3. Access the Streamlit Dashboard at `http://localhost:8501`
4. Access the MLflow Tracking Server at `http://localhost:5000`

### Method 2: Local Python Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate the environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Download Data: `python src/data/download_dataset.py`
5. Run Data Pipeline: `python src/data/make_dataset.py`
6. Run Feature Engineering: `python src/features/build_features.py`
7. Train Models: Run the scripts in `src/forecasting`, `src/segmentation`, and `src/churn`.
8. Start Dashboard: `streamlit run dashboard/app.py`

## 📊 Models & Explainability

- **XGBoost Churn Model:** Evaluated on Accuracy, Precision, Recall, F1, and ROC AUC. SHAP summary plots are generated automatically in `reports/churn/` for deep feature explainability.
- **K-Means Segmentation:** Optimal K determined via Silhouette analysis.
- **Prophet Forecaster:** Configured for strong weekly and yearly seasonality effects prevalent in retail.

## 🧪 Testing

Run tests using pytest:
```bash
pytest tests/
```
Ensure code coverage remains > 80% to maintain enterprise standards.

## 🤝 Contribution
Please open an issue to discuss proposed changes before submitting a Pull Request.
