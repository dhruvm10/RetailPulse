# RetailPulse: AI-Powered Retail Analytics Platform
**Comprehensive Project Report**

---

## 1. Executive Summary
**RetailPulse** is an end-to-end, enterprise-grade AI analytics platform designed to solve critical retail business challenges. It shifts the paradigm from simple descriptive dashboards to predictive and prescriptive machine learning. 

The platform leverages a real-world transactional dataset (Online Retail II) to deliver actionable intelligence across four primary domains:
1. **Customer Segmentation:** Identifying high-value and at-risk customers.
2. **Churn Prediction:** Proactively flagging customers likely to stop purchasing.
3. **Demand Forecasting:** Predicting future revenue streams.
4. **Inventory Optimization:** Calculating safety stock to prevent stockouts while minimizing holding costs.

---

## 2. System Architecture & Tech Stack
The project was designed with MLOps best practices and modularity in mind. 

### Technology Stack
- **Data Engineering:** Python, Pandas, NumPy
- **Machine Learning:** Scikit-Learn (K-Means), XGBoost, Prophet, SHAP
- **MLOps & Tracking:** MLflow (Experiment Tracking), Evidently AI (Data Drift Monitoring)
- **User Interface:** Streamlit, Plotly (Interactive Visualization)
- **Deployment:** Docker, Docker Compose, GitHub Actions (CI/CD)

### Pipeline Flow
1. **Data Extraction & Cleaning (`make_dataset.py`):** Raw Excel files are parsed. Nulls, duplicates, and cancelled orders are removed.
2. **Feature Engineering (`build_features.py`):** Calculates Recency, Frequency, Monetary (RFM) metrics for customers, and demand variances for products.
3. **Model Training (Modular Scripts):** ML models are trained independently and artifacts (`.pkl` files) are saved.
4. **Presentation (`app.py`):** The Streamlit dashboard dynamically loads these models and datasets based on centralized configurations.

---

## 3. Machine Learning Models & Methodologies

### 3.1. Customer Segmentation
- **Objective:** Group customers based on purchasing behavior to tailor marketing strategies.
- **Algorithm:** **K-Means Clustering**
- **Methodology:** Features used include Recency, Frequency, Monetary Value, and Average Order Value. The data was log-transformed to handle right-skewness and scaled using `StandardScaler`.
- **Evaluation:** Evaluated using the **Silhouette Score**, ensuring clusters are distinct and well-separated.

### 3.2. Customer Churn Prediction
- **Objective:** Predict which customers will likely not return (defined by a configurable threshold, default 90 days).
- **Algorithm:** **XGBoost Classifier**
- **Methodology:** A highly robust gradient boosting framework trained on the RFM features. 
- **Explainability:** **SHAP** (SHapley Additive exPlanations) is integrated to explain exactly *why* a customer was flagged as high risk (e.g., "Frequency is too low").
- **Evaluation:** Precision, Recall, F1-Score, and ROC-AUC.

### 3.3. Demand Forecasting
- **Objective:** Forecast daily revenue for the next 30 days to aid in financial planning.
- **Algorithm:** **Prophet** (by Meta)
- **Methodology:** Captures daily and weekly seasonality. Due to known optimization bugs with Prophet on certain Windows architectures, the system includes an automatic fallback to a **Simple Moving Average (SMA)**, ensuring the dashboard never crashes in production.

### 3.4. Inventory Optimization
- **Objective:** Calculate Reorder Points and Safety Stock.
- **Algorithm:** **Statistical Modeling**
- **Methodology:** Utilizes Lead Time Demand and Demand Variance. 
- **Formula:** `Safety Stock = Z-Score * Standard Deviation of Demand * sqrt(Lead Time)`
- **Impact:** Ensures highly volatile products have a buffer, while stable products don't waste warehouse space.

---

## 4. MLOps & Production Readiness

To ensure the platform is robust enough for enterprise deployment, several MLOps tools were integrated:
- **Centralized Configuration:** A `config.yaml` file dictates all hyperparameters and file paths, eliminating hardcoded variables.
- **MLflow:** Automatically tracks hyperparameters (e.g., number of XGBoost trees) and performance metrics (e.g., Accuracy) during every training run.
- **Evidently AI:** Generates automated HTML reports to detect "Data Drift"—alerting data scientists if the underlying statistical distribution of customer behavior changes over time.
- **CI/CD Pipeline:** A GitHub Actions workflow (`ci.yml`) automatically runs `flake8` linting and `pytest` integration tests on every code commit.

---

## 5. Conclusion & Business Impact
RetailPulse successfully bridges the gap between raw transactional data and strategic business action. By utilizing this platform, a retailer can:
- **Increase Revenue** through targeted marketing to segmented clusters.
- **Improve Retention** by intervening with high-churn-probability customers.
- **Reduce Costs** by mathematically optimizing inventory holding and reorder points.

The system is fully automated, thoroughly tested, and ready for continuous operation.
