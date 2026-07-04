import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# Configure page settings
st.set_page_config(page_title="RetailPulse | AI Retail Analytics", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for premium aesthetic
st.markdown("""
<style>
    .reportview-container {
        background: #f4f6f9;
    }
    .sidebar .sidebar-content {
        background: #2c3e50;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3498db;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_retail_data.csv')
CUSTOMER_SEGMENTS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'customer_segments.csv')
CHURN_PROBS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'customer_churn_probs.csv')
FORECAST_PATH = os.path.join(BASE_DIR, 'reports', 'forecasting', '30_day_forecast.csv')
INVENTORY_PATH = os.path.join(BASE_DIR, 'reports', 'inventory', 'inventory_recommendations.csv')

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def main():
    st.sidebar.title("🛒 RetailPulse")
    st.sidebar.markdown("### AI Powered Retail Analytics")
    
    menu = ["Executive Overview", "Sales Analytics", "Customer Segmentation", "Churn Prediction", "Demand Forecasting", "Inventory Optimization"]
    choice = st.sidebar.radio("Navigation", menu)
    
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

def show_executive_overview():
    st.title("Executive Overview")
    df = load_data(PROCESSED_DATA_PATH)
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${df['revenue'].sum():,.2f}")
        col2.metric("Total Orders", f"{df['invoice'].nunique():,}")
        col3.metric("Total Customers", f"{df['customer_id'].nunique():,}")
        col4.metric("Total Products", f"{df['stockcode'].nunique():,}")
        
        st.markdown("---")
        st.subheader("Top Revenue Generating Countries")
        country_rev = df.groupby('country')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False).head(10)
        fig = px.bar(country_rev, x='country', y='revenue', color='revenue', color_continuous_scale='Blues', title="Revenue by Country")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data not available. Please run the ETL pipeline first.")

def show_sales_analytics():
    st.title("Sales Analytics")
    df = load_data(PROCESSED_DATA_PATH)
    if df is not None:
        df['invoicedate'] = pd.to_datetime(df['invoicedate'])
        daily_sales = df.groupby(df['invoicedate'].dt.date)['revenue'].sum().reset_index()
        fig = px.line(daily_sales, x='invoicedate', y='revenue', title="Daily Revenue Trend", template='plotly_white')
        fig.update_traces(line_color='#2ecc71')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data not available.")

def show_customer_segmentation():
    st.title("Customer Segmentation")
    df = load_data(CUSTOMER_SEGMENTS_PATH)
    if df is not None:
        st.subheader("Customer Segments Overview")
        seg_counts = df['segment_name'].value_counts().reset_index()
        seg_counts.columns = ['segment_name', 'count']
        fig = px.pie(seg_counts, values='count', names='segment_name', title="Customer Distribution by Segment", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Segment Analysis (Recency vs Monetary)")
        fig2 = px.scatter(df, x='recency', y='monetary', color='segment_name', size='frequency', hover_data=['customer_id'], template='plotly_white', opacity=0.7)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Segmentation data not available. Please run the segmentation pipeline.")

def show_churn_prediction():
    st.title("Churn Prediction")
    df = load_data(CHURN_PROBS_PATH)
    if df is not None:
        st.subheader("High-Risk Customers")
        risk_df = df[df['churn_probability'] > 0.7].sort_values(by='churn_probability', ascending=False)
        st.dataframe(risk_df[['customer_id', 'recency', 'monetary', 'churn_probability']].head(20), use_container_width=True)
        
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(df, x="churn_probability", nbins=50, title="Distribution of Churn Probabilities", color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Churn data not available.")

def show_demand_forecasting():
    st.title("Demand Forecasting")
    df = load_data(FORECAST_PATH)
    if df is not None:
        st.subheader("30-Day Revenue Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat'], mode='lines', name='Predicted Revenue', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,255,255,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['ds'], y=df['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,255,255,0)', name='Confidence Interval', fillcolor='rgba(65, 105, 225, 0.2)'))
        fig.update_layout(title="Prophet Forecast", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Forecast data not available.")

def show_inventory_optimization():
    st.title("Inventory Optimization")
    df = load_data(INVENTORY_PATH)
    if df is not None:
        st.subheader("Reorder Recommendations")
        # Show top products that need reordering
        reorder_df = df[['stockcode', 'total_quantity_sold', 'safety_stock', 'reorder_point', 'reorder_quantity']].head(20)
        st.dataframe(reorder_df, use_container_width=True)
        
        fig = px.bar(reorder_df, x='stockcode', y=['safety_stock', 'reorder_point'], title="Safety Stock vs Reorder Point by Product", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Inventory data not available.")

if __name__ == "__main__":
    main()
