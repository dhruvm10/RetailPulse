import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_eda(input_path: str, report_dir: str):
    logger.info(f"Loading cleaned data from {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        return
        
    df = pd.read_csv(input_path)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    
    os.makedirs(report_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # 1. Sales Analysis (Daily, Monthly)
    df.set_index('invoicedate', inplace=True)
    daily_sales = df['revenue'].resample('D').sum()
    monthly_sales = df['revenue'].resample('ME').sum()
    df.reset_index(inplace=True)
    
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='bar', color='skyblue')
    plt.title('Monthly Sales Revenue')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'monthly_sales.png'))
    plt.close()
    
    # 2. Customer Analysis
    customer_revenue = df.groupby('customer_id')['revenue'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.histplot(customer_revenue[customer_revenue < 10000], bins=50, kde=True, color='purple')
    plt.title('Customer Lifetime Value Distribution (Revenue < 10k)')
    plt.xlabel('Revenue')
    plt.ylabel('Count')
    plt.savefig(os.path.join(report_dir, 'customer_ltv_dist.png'))
    plt.close()
    
    # 3. Product Analysis
    top_products = df.groupby('description')['quantity'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
    plt.title('Top 10 Products by Quantity Sold')
    plt.xlabel('Total Quantity')
    plt.ylabel('Product Description')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'top_products.png'))
    plt.close()
    
    # 4. Country Analysis
    top_countries = df.groupby('country')['revenue'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 8))
    plt.pie(top_countries.values, labels=top_countries.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Revenue by Top 10 Countries')
    plt.savefig(os.path.join(report_dir, 'revenue_by_country.png'))
    plt.close()
    
    # 5. Correlation Matrix
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(report_dir, 'correlation_matrix.png'))
    plt.close()
    
    # Generate EDA Report Summary
    report_content = f"""# Exploratory Data Analysis Report

## Summary Statistics
- **Total Revenue:** {df['revenue'].sum():.2f}
- **Total Orders:** {df['invoice'].nunique()}
- **Unique Customers:** {df['customer_id'].nunique()}
- **Unique Products:** {df['stockcode'].nunique()}

## Visualizations Generated
1. Monthly Sales (`monthly_sales.png`)
2. Customer LTV Distribution (`customer_ltv_dist.png`)
3. Top 10 Products by Quantity (`top_products.png`)
4. Revenue by Country (`revenue_by_country.png`)
5. Correlation Matrix (`correlation_matrix.png`)
"""
    
    with open(os.path.join(report_dir, 'eda_summary.md'), "w") as f:
        f.write(report_content)
    logger.info(f"EDA complete. Charts and report saved to {report_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, 'data', 'processed', 'cleaned_retail_data.csv')
    report_dir = os.path.join(base_dir, 'reports', 'eda')
    perform_eda(input_file, report_dir)
