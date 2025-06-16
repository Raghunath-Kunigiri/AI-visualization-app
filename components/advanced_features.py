import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px

def create_preprocessing_section(df):
    """Advanced data preprocessing options"""
    if df is None:
        return df
    
    st.subheader("🔧 Data Preprocessing")
    
    # Missing values handling
    if df.isnull().sum().sum() > 0:
        st.write("**Missing Values Found:**")
        missing_strategy = st.selectbox(
            "Choose missing value strategy:",
            ["Keep as is", "Drop rows", "Fill with mean", "Fill with median", "Forward fill"]
        )
        
        if missing_strategy != "Keep as is":
            if missing_strategy == "Drop rows":
                df = df.dropna()
            elif missing_strategy == "Fill with mean":
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_strategy == "Fill with median":
                df = df.fillna(df.median(numeric_only=True))
            elif missing_strategy == "Forward fill":
                df = df.fillna(method='ffill')
            
            st.success(f"✅ Applied {missing_strategy}")
    
    # Outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        detect_outliers = st.checkbox("🎯 Detect and highlight outliers")
        if detect_outliers:
            outlier_method = st.selectbox("Outlier detection method:", ["IQR", "Z-Score"])
            selected_col = st.selectbox("Select column for outlier detection:", numeric_cols)
            
            if outlier_method == "IQR":
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[selected_col] < (Q1 - 1.5 * IQR)) | 
                             (df[selected_col] > (Q3 + 1.5 * IQR))]
            else:  # Z-Score
                z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
                outliers = df[z_scores > 3]
            
            st.write(f"**Found {len(outliers)} outliers in {selected_col}:**")
            if len(outliers) > 0:
                st.dataframe(outliers[[selected_col]])
                
                if st.button("🗑️ Remove outliers"):
                    df = df.drop(outliers.index)
                    st.success(f"✅ Removed {len(outliers)} outliers")
    
    return df

def create_advanced_models_section():
    """Create controls for advanced ML models"""
    st.subheader("🎯 Advanced Models")
    
    advanced_models = ['Support Vector Machine', 'Neural Network']
    selected_advanced_model = st.selectbox("Select Advanced Model", advanced_models)
    
    model_params = {}
    
    if selected_advanced_model == 'Support Vector Machine':
        model_params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
        model_params['C'] = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
        
    elif selected_advanced_model == 'Neural Network':
        model_params['hidden_layers'] = st.slider("Hidden Layers", 1, 3, 2)
        model_params['neurons'] = st.slider("Neurons per layer", 10, 200, 100)
        model_params['max_iter'] = st.slider("Max iterations", 100, 1000, 500)
    
    return selected_advanced_model, model_params

def train_advanced_model(df, x_axis, y_axis, model_name, model_params, test_size, random_state):
    """Train advanced ML models"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare data
    X = df[[x_axis]]
    y = df[y_axis]
    
    # Scale features for neural networks and SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Create model
    if model_name == 'Support Vector Machine':
        model = SVR(
            kernel=model_params['kernel'],
            C=model_params['C']
        )
    elif model_name == 'Neural Network':
        hidden_layer_sizes = tuple([model_params['neurons']] * model_params['hidden_layers'])
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=model_params['max_iter'],
            random_state=random_state
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    return {
        'model': model,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'test_pred': test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'cv_scores': cv_scores
    }

def create_feature_engineering_section(df):
    """Feature engineering options"""
    if df is None:
        return df
    
    st.subheader("⚗️ Feature Engineering")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        eng_options = st.multiselect(
            "Select feature engineering options:",
            ["Log transform", "Square root", "Polynomial features", "Standardization"]
        )
        
        if eng_options:
            feature_col = st.selectbox("Select column for engineering:", numeric_cols)
            
            for option in eng_options:
                if option == "Log transform":
                    if (df[feature_col] > 0).all():
                        df[f'{feature_col}_log'] = np.log(df[feature_col])
                        st.success(f"✅ Added {feature_col}_log")
                    else:
                        st.warning("⚠️ Log transform requires positive values")
                
                elif option == "Square root":
                    if (df[feature_col] >= 0).all():
                        df[f'{feature_col}_sqrt'] = np.sqrt(df[feature_col])
                        st.success(f"✅ Added {feature_col}_sqrt")
                    else:
                        st.warning("⚠️ Square root requires non-negative values")
                
                elif option == "Polynomial features":
                    degree = st.slider("Polynomial degree", 2, 4, 2)
                    df[f'{feature_col}_poly{degree}'] = df[feature_col] ** degree
                    st.success(f"✅ Added {feature_col}_poly{degree}")
                
                elif option == "Standardization":
                    scaler = StandardScaler()
                    df[f'{feature_col}_scaled'] = scaler.fit_transform(df[[feature_col]])
                    st.success(f"✅ Added {feature_col}_scaled")
    
    return df

def create_model_comparison_dashboard(results_dict):
    """Create a dashboard comparing multiple models"""
    if not results_dict:
        return
    
    st.subheader("📊 Model Comparison Dashboard")
    
    # Create comparison metrics dataframe
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Test R²': results['test_r2'],
            'Train R²': results['train_r2'],
            'Test MSE': results['test_mse'],
            'CV Mean R²': results.get('cv_scores', [0]).mean() if 'cv_scores' in results else 0
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display metrics table
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_r2 = px.bar(comparison_df, x='Model', y='Test R²', 
                       title='Model Performance (R² Score)',
                       color='Test R²',
                       color_continuous_scale='viridis')
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        fig_mse = px.bar(comparison_df, x='Model', y='Test MSE', 
                        title='Model Error (MSE)',
                        color='Test MSE',
                        color_continuous_scale='reds')
        st.plotly_chart(fig_mse, use_container_width=True)

def create_data_quality_report(df):
    """Generate comprehensive data quality report"""
    if df is None:
        return
    
    st.subheader("📋 Data Quality Report")
    
    quality_metrics = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Memory Usage (MB)': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Display quality metrics
    cols = st.columns(len(quality_metrics))
    for i, (metric, value) in enumerate(quality_metrics.items()):
        with cols[i]:
            st.metric(metric, f"{value:.2f}" if isinstance(value, float) else value)
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    fig_dtypes = px.pie(values=dtype_counts.values, names=[str(dtype) for dtype in dtype_counts.index],
                       title="Data Types Distribution")
    st.plotly_chart(fig_dtypes, use_container_width=True)
    
    # Missing values heatmap
    if df.isnull().sum().sum() > 0:
        st.write("**Missing Values Pattern:**")
        missing_matrix = df.isnull().astype(int)
        fig_missing = px.imshow(missing_matrix.T, 
                               title="Missing Values Heatmap",
                               labels=dict(x="Row Index", y="Columns"))
        st.plotly_chart(fig_missing, use_container_width=True) 