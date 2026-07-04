echo "Starting RetailPulse Pipeline..."

echo "1. Downloading Dataset..."
python src/data/download_dataset.py

echo "2. Cleaning Data..."
python src/data/make_dataset.py

echo "3. Engineering Features..."
python src/features/build_features.py

echo "4. Running EDA..."
python src/features/eda_profiling.py

echo "5. Training Segmentation Model..."
python src/segmentation/kmeans_model.py

echo "6. Training Forecasting Model..."
python src/forecasting/train_forecaster.py

echo "7. Training Churn Prediction Model..."
python src/churn/churn_model.py

echo "8. Running Inventory Optimization..."
python src/inventory/inventory_engine.py

echo "Pipeline Complete! You can now start the dashboard:"
echo "streamlit run dashboard/app.py"
