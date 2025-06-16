# import streamlit as st  
# import pandas as pd
# from sklearn.linear_model  import LinearRegression
# from sklearn.model_selection import train_test_split
# import plotly.express as px


# st.title("📊 Smart Data Predictor & Visualizer")

# # Upload CSV
# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("📄 Raw Data")
#     st.write(df)

#     # Column Selection
#     numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
#     st.sidebar.header("🔧 Controls")
#     x_axis = st.sidebar.selectbox("Select X-axis (Feature)", numeric_columns)
#     y_axis = st.sidebar.selectbox("Select Y-axis (Target)", numeric_columns)

#     # Choose Chart
#     chart_type = st.sidebar.radio("Select Chart Type", ['Line', 'Scatter', 'Bar'])

#     # Visualize
#     if st.button("Generate Chart"):
#         st.subheader("📈 Data Visualization")
#         if chart_type == 'Line':
#             fig = px.line(df, x=x_axis, y=y_axis, title=f"{chart_type} Chart")
#         elif chart_type == 'Scatter':
#             fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{chart_type} Chart")
#         elif chart_type == 'Bar':
#             fig = px.bar(df, x=x_axis, y=y_axis, title=f"{chart_type} Chart")
#         st.plotly_chart(fig)

#     # Train model
#     if st.button("Train AI Model and Predict"):
#         X = df[[x_axis]]
#         y = df[y_axis]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         model = LinearRegression()
#         model.fit(X_train, y_train)

#         predictions = model.predict(X_test)

#         result_df = pd.DataFrame({x_axis: X_test[x_axis], "Actual": y_test, "Predicted": predictions})
#         st.subheader("🤖 Prediction Results")
#         st.write(result_df)

#         fig2 = px.scatter(result_df, x=x_axis, y=["Actual", "Predicted"], title="Actual vs Predicted")
#         st.plotly_chart(fig2)









import streamlit as st
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from components.uploader import upload_csv, get_numeric_columns, display_data_summary
from components.visualizer import create_chart_controls, generate_chart, display_correlation_matrix
from components.model_trainer import create_model_controls, train_and_evaluate_model, display_model_results
from components.advanced_features import (
    create_preprocessing_section, create_advanced_models_section, 
    train_advanced_model, create_feature_engineering_section,
    create_model_comparison_dashboard, create_data_quality_report
)
from components.api_integration import (
    create_api_data_section, create_real_time_dashboard, 
    create_data_streaming_simulation
)
from utils.helpers import toggle_theme, display_data_summary as detailed_summary, create_download_section, add_footer

# Page configuration
st.set_page_config(
    page_title="Smart Data Predictor & Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styling.")

load_css()

# Main app title
st.markdown("""
<div class="app-header">
    <h1 class="app-title">📊 Smart Data Predictor & Visualizer</h1>
    <p class="app-subtitle">Advanced Analytics with Multiple ML Models & Real-time Data</p>
</div>
""", unsafe_allow_html=True)

# Theme toggle
toggle_theme()

def main():
    """Main application logic"""
    
    # Data source selection
    st.header("📁 Data Source")
    data_source = st.radio(
        "Choose your data source:",
        ["📤 Upload CSV", "🌐 Real-time API Data"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "📤 Upload CSV":
        df = upload_csv()
    else:
        df = create_api_data_section()
    
    if df is not None:
        # Display basic data summary
        display_data_summary(df)
        
        # Get numeric columns
        numeric_columns = get_numeric_columns(df)
        
        if len(numeric_columns) >= 2:
            # Create enhanced tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📈 Visualization", "🤖 Model Training", "🔧 Advanced Features", 
                "🌐 Real-time Dashboard", "📋 Data Analysis", "💾 Export"
            ])
            
            # Visualization Tab
            with tab1:
                st.header("📈 Data Visualization")
                
                x_axis, y_axis, chart_type = create_chart_controls(numeric_columns)
                
                if x_axis and y_axis and chart_type:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if st.button("🎨 Generate Chart", type="primary"):
                            fig = generate_chart(df, x_axis, y_axis, chart_type)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.info("💡 **Tips:**\n\n"
                               "• Try different chart types\n"
                               "• Look for patterns and trends\n"
                               "• Check correlation matrix below")
                
                # Correlation matrix
                display_correlation_matrix(df, numeric_columns)
            
            # Model Training Tab
            with tab2:
                st.header("🤖 Machine Learning Models")
                
                # Model controls
                selected_model, test_size, random_state = create_model_controls()
                
                # Feature selection for model
                st.subheader("🎯 Feature Selection")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_feature = st.selectbox("Select Feature (X)", numeric_columns, key="model_x")
                with col2:
                    y_target = st.selectbox("Select Target (y)", numeric_columns, key="model_y")
                
                if x_feature and y_target and x_feature != y_target:
                    if st.button("🚀 Train Model", type="primary"):
                        with st.spinner(f"Training {selected_model} model..."):
                            results = train_and_evaluate_model(
                                df, x_feature, y_target, selected_model, test_size, random_state
                            )
                            
                            if results:
                                display_model_results(results, x_feature, y_target, selected_model)
                else:
                    st.warning("⚠️ Please select different features for X and y")
            
            # Advanced Features Tab
            with tab3:
                st.header("🔧 Advanced Features")
                
                # Data preprocessing
                df_processed = create_preprocessing_section(df.copy())
                
                # Feature engineering
                if df_processed is not None:
                    df_processed = create_feature_engineering_section(df_processed)
                
                # Advanced models
                st.divider()
                advanced_model, model_params = create_advanced_models_section()
                
                if st.button("🎯 Train Advanced Model", type="primary"):
                    if df_processed is not None and len(get_numeric_columns(df_processed)) >= 2:
                        adv_numeric_cols = get_numeric_columns(df_processed)
                        x_adv = st.selectbox("Select X feature:", adv_numeric_cols, key="adv_x")
                        y_adv = st.selectbox("Select y target:", adv_numeric_cols, key="adv_y")
                        
                        if x_adv != y_adv:
                            with st.spinner(f"Training {advanced_model}..."):
                                adv_results = train_advanced_model(
                                    df_processed, x_adv, y_adv, advanced_model, 
                                    model_params, test_size, random_state
                                )
                                
                                if adv_results:
                                    st.success(f"✅ {advanced_model} trained successfully!")
                                    
                                    # Display metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Test R²", f"{adv_results['test_r2']:.4f}")
                                    with col2:
                                        st.metric("Cross-Val R²", f"{adv_results['cv_scores'].mean():.4f}")
                                    with col3:
                                        st.metric("CV Std", f"{adv_results['cv_scores'].std():.4f}")
                
                # Data quality report
                st.divider()
                create_data_quality_report(df_processed if df_processed is not None else df)
            
            # Real-time Dashboard Tab
            with tab4:
                st.header("🌐 Real-time Dashboard")
                
                # Real-time dashboard for time series data
                create_real_time_dashboard(df)
                
                # Data streaming simulation
                st.divider()
                create_data_streaming_simulation()
            
            # Data Analysis Tab
            with tab5:
                st.header("📋 Detailed Data Analysis")
                detailed_summary(df)
            
            # Export Tab
            with tab6:
                st.header("💾 Export Options")
                create_download_section(df)
        
        else:
            st.error("❌ Need at least 2 numeric columns for analysis. Please upload a dataset with numeric data.")
    
    else:
        # Show enhanced instructions when no data is uploaded
        st.markdown("""
        ### 🚀 Welcome to Smart Data Predictor & Visualizer!
        
        **🆕 New Features Added:**
        - 🤖 **Advanced ML Models**: Support Vector Machines, Neural Networks
        - 🔧 **Data Preprocessing**: Handle missing values, outliers, feature engineering
        - 🌐 **Real-time Data**: Connect to APIs, streaming data simulation
        - 📊 **Enhanced Analytics**: Model comparison, data quality reports
        - 🎨 **Better UI**: Improved styling and user experience
        
        **To get started:**
        1. 📤 Upload a CSV file OR connect to real-time data
        2. 🔍 Explore with advanced visualizations
        3. 🤖 Train multiple ML models and compare results
        4. 🔧 Use advanced preprocessing and feature engineering
        5. 📊 Analyze with comprehensive reports
        
        **What's New:**
        - 🎯 SVM and Neural Network models
        - ⚗️ Automatic feature engineering
        - 🔍 Outlier detection and removal
        - 📈 Model comparison dashboard
        - 🌊 Real-time data streaming
        - 📋 Data quality assessments
        """)
        
        # Enhanced sample data section
        with st.expander("📋 Need sample data? Choose your dataset!", expanded=False):
            sample_type = st.selectbox(
                "Select sample dataset type:",
                ["Financial Data", "Weather Data", "Sensor Data", "Random Dataset"]
            )
            
            if st.button(f"🎲 Generate {sample_type}"):
                if sample_type == "Financial Data":
                    df = create_api_data_section() if st.selectbox("Data source:", ["Sample Stock Data"]) == "Sample Stock Data" else None
                else:
                    import pandas as pd
                    import numpy as np
                    
                    # Generate sample data based on type
                    np.random.seed(42)
                    n_samples = 200
                    
                    if sample_type == "Weather Data":
                        df = pd.DataFrame({
                            'temperature': np.random.normal(20, 10, n_samples),
                            'humidity': np.random.uniform(30, 90, n_samples),
                            'pressure': np.random.normal(1013, 15, n_samples),
                            'wind_speed': np.random.exponential(3, n_samples)
                        })
                    elif sample_type == "Sensor Data":
                        time_series = np.cumsum(np.random.randn(n_samples)) * 0.1
                        df = pd.DataFrame({
                            'sensor_1': time_series + np.random.normal(0, 0.1, n_samples),
                            'sensor_2': time_series * 1.5 + np.random.normal(0, 0.2, n_samples),
                            'sensor_3': np.random.uniform(0, 100, n_samples),
                            'quality_score': np.random.beta(2, 2, n_samples) * 100
                        })
                    else:  # Random Dataset
                        df = pd.DataFrame({
                            'feature_1': np.random.normal(50, 15, n_samples),
                            'feature_2': np.random.exponential(2, n_samples),
                            'target': np.random.normal(100, 20, n_samples),
                            'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
                        })
                    
                    st.session_state['sample_data'] = df
                    st.success(f"✅ {sample_type} generated! Scroll up to start analyzing.")
                    st.dataframe(df.head(10))

    # Enhanced footer
    add_footer()
    
    # Additional footer with new features
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
    🆕 <strong>Latest Updates:</strong> Advanced ML Models • Real-time Data • Enhanced Preprocessing<br>
    💡 <strong>Next:</strong> Deploy to cloud, add more APIs, create mobile app
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 