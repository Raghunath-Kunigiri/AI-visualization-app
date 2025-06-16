import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Try to import XGBoost with fallback
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def create_model_controls():
    """Create controls for model selection and training"""
    st.sidebar.header("🤖 Model Training")
    
    available_models = ['Linear Regression', 'Random Forest']
    if XGBOOST_AVAILABLE:
        available_models.append('XGBoost')
    
    selected_model = st.sidebar.selectbox("Select Model", available_models)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", value=42, min_value=0)
    
    return selected_model, test_size, random_state

def get_model(model_name):
    """Get model instance based on name"""
    if model_name == 'Linear Regression':
        return LinearRegression()
    elif model_name == 'Random Forest':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
        return XGBRegressor(n_estimators=100, random_state=42)
    else:
        st.error(f"Model {model_name} not available")
        return None

def train_and_evaluate_model(df, x_axis, y_axis, model_name, test_size, random_state):
    """Train model and return results"""
    if df is None or x_axis is None or y_axis is None:
        return None
    
    try:
        # Prepare data
        X = df[[x_axis]]
        y = df[y_axis]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Get and train model
        model = get_model(model_name)
        if model is None:
            return None
        
        model.fit(X_train, y_train)
        
        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        
        results = {
            'model': model,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'metrics': {
                'train_mse': train_mse, 'test_mse': test_mse,
                'train_r2': train_r2, 'test_r2': test_r2,
                'train_mae': train_mae, 'test_mae': test_mae
            }
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def display_model_results(results, x_axis, y_axis, model_name):
    """Display model training results"""
    if results is None:
        return
    
    st.subheader(f"🤖 {model_name} Results")
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Metrics:**")
        st.metric("R² Score", f"{results['metrics']['train_r2']:.4f}")
        st.metric("MSE", f"{results['metrics']['train_mse']:.4f}")
        st.metric("MAE", f"{results['metrics']['train_mae']:.4f}")
    
    with col2:
        st.write("**Test Metrics:**")
        st.metric("R² Score", f"{results['metrics']['test_r2']:.4f}")
        st.metric("MSE", f"{results['metrics']['test_mse']:.4f}")
        st.metric("MAE", f"{results['metrics']['test_mae']:.4f}")
    
    # Create predictions dataframe
    result_df = pd.DataFrame({
        x_axis: results['X_test'][x_axis],
        "Actual": results['y_test'],
        "Predicted": results['test_predictions']
    })
    
    st.subheader("📊 Prediction Results")
    st.write(result_df)
    
    # Plot actual vs predicted
    fig = px.scatter(result_df, 
                    x="Actual", 
                    y="Predicted", 
                    title=f"Actual vs Predicted - {model_name}",
                    trendline="ols")
    
    # Add perfect prediction line
    min_val = min(result_df["Actual"].min(), result_df["Predicted"].min())
    max_val = max(result_df["Actual"].max(), result_df["Predicted"].max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], 
                           y=[min_val, max_val],
                           mode='lines',
                           name='Perfect Prediction',
                           line=dict(dash='dash', color='red')))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance for tree-based models
    if hasattr(results['model'], 'feature_importances_'):
        st.subheader("📈 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': [x_axis],
            'Importance': results['model'].feature_importances_
        })
        fig_imp = px.bar(importance_df, x='Feature', y='Importance', 
                        title="Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True) 