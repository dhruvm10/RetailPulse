import streamlit as st
import pandas as pd
import os
import yaml
import plotly.express as px
import plotly.graph_objects as go
import io

# Setup Page
st.set_page_config(page_title="RetailPulse | Enterprise Retail Intelligence", page_icon="📈", layout="wide")

st.markdown("""
<style>
    /* Premium Dark Mode / Glassmorphism Styling */
    .stApp {
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* High contrast text for headers */
    h1, h2, h3 { 
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Premium KPI Cards (Glassmorphism) */
    [data-testid="stMetric"], [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.9) !important;
        padding: 20px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-left: 6px solid #00F0FF !important;
        transition: transform 0.2s ease-in-out !important;
    }
    
    [data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 240, 255, 0.2) !important;
    }
    
    /* Make metric text pop */
    [data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] label {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    /* Premium Buttons */
    .stButton>button { 
        border-radius: 8px; 
        font-weight: 600; 
        background: linear-gradient(135deg, #00F0FF 0%, #0080FF 100%);
        color: white !important;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 128, 255, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0080FF 0%, #00F0FF 100%);
        box-shadow: 0 6px 20px rgba(0, 240, 255, 0.5);
        transform: translateY(-2px);
    }
    
    /* Dataframes and Tables Contrast */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Backend Integration
@st.cache_resource
def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config, base_dir

config, BASE_DIR = load_config()

# Data Loaders with Error Handling
@st.cache_data
def load_data(relative_path):
    path = os.path.join(BASE_DIR, relative_path)
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load data from {relative_path}: {e}")
        return None

def download_button(df, filename, label="Download CSV"):
    csv = df.to_csv(index=False)
    st.download_button(label=label, data=csv, file_name=filename, mime='text/csv')

def main():
    st.sidebar.title("📈 RetailPulse")
    st.sidebar.markdown("**AI Powered Retail Analytics**")
    st.sidebar.markdown("---")
    
    menu = ["Executive Overview", "Sales Analytics", "Customer Segmentation", "Churn Prediction", "Demand Forecasting", "Inventory Optimization"]
    choice = st.sidebar.radio("Navigation", menu)
    
    st.sidebar.markdown("---")
    st.sidebar.info("RetailPulse leverages Prophet, K-Means, XGBoost, and SHAP to deliver enterprise intelligence.")
    
    try:
        if choice == "Executive Overview":
            show_executive_overview()
        elif choice == "Sales Analytics":
            show_sales_analytics()
        elif choice == "Customer Segmentation":
            show_customer_segmentation()
        elif choice == "Churn Prediction":
            show_churn_prediction()
        elif choice == "Demand Forecasting":
            show_demand_forecasting()
        elif choice == "Inventory Optimization":
            show_inventory_optimization()
    except Exception as e:
        st.error("An unexpected error occurred while rendering the dashboard.")
        st.exception(e)

def show_executive_overview():
    st.title("Executive Overview")
    df = load_data(config['paths']['processed_data'])
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
        col2.metric("Total Orders", f"{df['invoice'].nunique():,}")
        col3.metric("Total Customers", f"{df['customer_id'].nunique():,}")
        col4.metric("Total Products", f"{df['stockcode'].nunique():,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        country_rev = df.groupby('country')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False).head(10)
        fig = px.bar(country_rev, x='country', y='revenue', color='revenue', color_continuous_scale='Tealgrn', title="Top 10 Regions by Revenue")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Processed dataset not found. Please run the ETL pipeline.")

def show_sales_analytics():
    st.title("Sales Analytics")
    df = load_data(config['paths']['processed_data'])
    
    if df is not None:
        df['invoicedate'] = pd.to_datetime(df['invoicedate'])
        daily_sales = df.groupby(df['invoicedate'].dt.date)['revenue'].sum().reset_index()
        
        fig = px.line(daily_sales, x='invoicedate', y='revenue', title="Daily Revenue Trend", template='plotly_white')
        fig.update_traces(line_color='#2ecc71', line_width=2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly Heatmap
        df['month'] = df['invoicedate'].dt.month
        df['day_of_week'] = df['invoicedate'].dt.day_name()
        heatmap_data = df.groupby(['day_of_week', 'month'])['revenue'].sum().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='month', values='revenue')
        
        fig_heat = px.imshow(heatmap_pivot, color_continuous_scale='Blues', title="Revenue Heatmap (Day vs Month)")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Data not available.")

def show_customer_segmentation():
    st.title("Customer Segmentation")
    df = load_data(config['paths']['segments'])
    
    if df is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            seg_counts = df['segment_name'].value_counts().reset_index()
            seg_counts.columns = ['segment_name', 'count']
            fig = px.pie(seg_counts, values='count', names='segment_name', title="Segment Distribution", hole=0.5, color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig2 = px.scatter(df, x='recency', y='monetary', color='segment_name', size='frequency', 
                              hover_data=['customer_id'], template='plotly_white', opacity=0.8,
                              title="RFM Scatter Plot (Recency vs Monetary)")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("Raw Segment Data")
        st.dataframe(df.head(100), use_container_width=True)
        download_button(df, "customer_segments.csv", "Download Full Segments Report")
    else:
        st.warning("Segmentation data not available. Please run the KMeans model.")

def show_churn_prediction():
    st.title("Churn Prediction")
    df = load_data(config['paths']['churn_probs'])
    
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="churn_probability", nbins=40, title="Churn Probability Distribution", color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("High-Risk Customer List")
            risk_df = df[df['churn_probability'] > 0.8].sort_values(by='churn_probability', ascending=False)
            st.dataframe(risk_df[['customer_id', 'recency', 'monetary', 'churn_probability']].head(15), use_container_width=True)
            
        st.markdown("---")
        download_button(risk_df, "high_risk_customers.csv", "Download High-Risk Customers CSV")
    else:
        st.warning("Churn data not available.")

def show_demand_forecasting():
    st.title("Demand Forecasting")
    forecast_path = os.path.join(config['paths']['reports_dir'], 'forecasting', '30_day_forecast.csv')
    df = load_data(forecast_path)
    
    if df is not None:
        st.subheader("30-Day Revenue Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat'], mode='lines', name='Predicted Revenue', line=dict(color='#3498db', width=3)))
        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,255,255,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,255,255,0)', name='Confidence Interval', fillcolor='rgba(52, 152, 219, 0.2)'))
        fig.update_layout(template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        download_button(df, "revenue_forecast.csv", "Download Forecast CSV")
    else:
        st.warning("Forecast data not available.")

def show_inventory_optimization():
    st.title("Inventory Optimization Engine")
    inv_path = os.path.join(config['paths']['reports_dir'], 'inventory', 'inventory_recommendations.csv')
    df = load_data(inv_path)
    
    if df is not None:
        st.info(f"Inventory calculated using a lead time of {config['inventory']['lead_time_days']} days.")
        
        reorder_df = df[['stockcode', 'total_quantity_sold', 'safety_stock', 'reorder_point', 'reorder_quantity']].sort_values(by='total_quantity_sold', ascending=False)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.bar(reorder_df.head(20), x='stockcode', y=['safety_stock', 'reorder_point'], 
                         title="Safety Stock vs Reorder Point (Top 20 Products)", barmode='group',
                         color_discrete_sequence=['#f39c12', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Action Required")
            st.markdown("These products require immediate attention based on lead time demand variance.")
            st.dataframe(reorder_df.head(10)[['stockcode', 'reorder_quantity']])
            
        download_button(reorder_df, "inventory_recommendations.csv", "Download Full Inventory Report")
    else:
        st.warning("Inventory data not available.")

if __name__ == "__main__":
    main()
