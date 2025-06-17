import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_weather_prediction_section(df):
    """Create weather prediction interface"""
    if df is None:
        st.warning("⚠️ Please load weather data first to make predictions")
        return None
    
    st.header("🌦️ Weather Prediction System")
    
    # Check if we have the required columns
    required_cols = ['temperature', 'humidity', 'pressure']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Weather data must contain temperature, humidity, and pressure columns")
        return None
    
    # Prepare weather data for prediction
    prediction_data = prepare_weather_data(df)
    
    if prediction_data is not None:
        # Create prediction tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "☔ Rain Prediction", "💨 Wind Prediction", 
            "☀️ Sunny Prediction", "🌪️ Severe Weather"
        ])
        
        with tab1:
            create_rain_predictor(prediction_data)
        
        with tab2:
            create_wind_predictor(prediction_data)
        
        with tab3:
            create_sunny_predictor(prediction_data)
        
        with tab4:
            create_severe_weather_predictor(prediction_data)

def prepare_weather_data(df):
    """Prepare and engineer features for weather prediction"""
    try:
        # Make a copy of the data
        data = df.copy()
        
        # Feature engineering
        st.subheader("📊 Weather Data Preparation")
        
        # Create derived features
        if 'temperature' in data.columns and 'humidity' in data.columns:
            # Heat index calculation
            data['heat_index'] = calculate_heat_index(data['temperature'], data['humidity'])
            
            # Dew point calculation
            data['dew_point'] = calculate_dew_point(data['temperature'], data['humidity'])
            
            # Temperature categories
            data['temp_category'] = pd.cut(data['temperature'], 
                                         bins=[-50, 0, 10, 20, 30, 100], 
                                         labels=['Freezing', 'Cold', 'Cool', 'Warm', 'Hot'])
        
        # Pressure trend (if we have time series data)
        if len(data) > 1:
            data['pressure_change'] = data['pressure'].diff()
            data['pressure_trend'] = np.where(data['pressure_change'] > 2, 'Rising',
                                             np.where(data['pressure_change'] < -2, 'Falling', 'Stable'))
        
        # Create synthetic weather conditions for training (since API doesn't provide this)
        data = create_synthetic_weather_labels(data)
        
        # Display data summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Data Points", len(data))
            st.metric("🌡️ Avg Temperature", f"{data['temperature'].mean():.1f}°C")
        with col2:
            st.metric("💧 Avg Humidity", f"{data['humidity'].mean():.1f}%")
            st.metric("🏔️ Avg Pressure", f"{data['pressure'].mean():.1f} hPa")
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error preparing weather data: {str(e)}")
        return None

def calculate_heat_index(temp, humidity):
    """Calculate heat index from temperature and humidity"""
    # Convert Celsius to Fahrenheit for calculation
    temp_f = (temp * 9/5) + 32
    
    # Heat index formula
    hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
    
    # Convert back to Celsius
    return (hi - 32) * 5/9

def calculate_dew_point(temp, humidity):
    """Calculate dew point from temperature and humidity"""
    a = 17.27
    b = 237.7
    
    alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    
    return dew_point

def create_synthetic_weather_labels(data):
    """Create synthetic weather condition labels for training"""
    # Rain probability based on humidity, pressure, and temperature
    rain_prob = (
        (data['humidity'] > 70) * 0.4 +
        (data['pressure'] < 1010) * 0.3 +
        (data['temperature'] > 15) * 0.2 +
        np.random.random(len(data)) * 0.1
    )
    data['will_rain'] = (rain_prob > 0.6).astype(int)
    
    # Wind prediction based on pressure changes and conditions
    if 'pressure_change' in data.columns:
        wind_prob = (
            (abs(data['pressure_change'].fillna(0)) > 3) * 0.5 +
            (data['temperature'] > 25) * 0.2 +
            np.random.random(len(data)) * 0.3
        )
    else:
        wind_prob = np.random.random(len(data))
    data['will_be_windy'] = (wind_prob > 0.5).astype(int)
    
    # Sunny prediction (inverse of rain + low humidity)
    sunny_prob = (
        (data['humidity'] < 50) * 0.4 +
        (data['pressure'] > 1015) * 0.3 +
        (data['temperature'] > 20) * 0.2 +
        (1 - rain_prob) * 0.1
    )
    data['will_be_sunny'] = (sunny_prob > 0.6).astype(int)
    
    # Severe weather (tornado/storm) prediction
    severe_prob = (
        (data['temperature'] > 25) * 0.2 +
        (data['humidity'] > 80) * 0.3 +
        (data['pressure'] < 1005) * 0.4 +
        np.random.random(len(data)) * 0.1
    )
    data['severe_weather'] = (severe_prob > 0.8).astype(int)
    
    return data

def create_rain_predictor(data):
    """Create rain prediction model and interface"""
    st.subheader("☔ Rain Prediction Model")
    
    # Features for rain prediction
    feature_cols = ['temperature', 'humidity', 'pressure', 'heat_index', 'dew_point']
    available_features = [col for col in feature_cols if col in data.columns]
    
    if len(available_features) < 3:
        st.error("❌ Not enough features for rain prediction")
        return
    
    # Prepare data
    X = data[available_features].fillna(data[available_features].mean())
    y = data['will_rain']
    
    # Train model
    model, accuracy, report = train_weather_model(X, y, "Rain")
    
    # Display model performance
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': available_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(importance_df, x='importance', y='feature', 
                        title="🔍 Feature Importance for Rain Prediction",
                        labels={'importance': 'Importance Score'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Make predictions for next days
        st.write("**📅 Rain Forecast:**")
        forecast = make_weather_forecast(model, data, available_features, 'rain')
        
        for i, (date, prob) in enumerate(forecast.items()):
            rain_emoji = "☔" if prob > 0.6 else "🌤️" if prob > 0.3 else "☀️"
            st.write(f"{rain_emoji} {date}: {prob:.0%} chance of rain")
    
    # Interactive prediction
    st.divider()
    create_interactive_rain_predictor(model, available_features)

def create_wind_predictor(data):
    """Create wind prediction model and interface"""
    st.subheader("💨 Wind Prediction Model")
    
    feature_cols = ['temperature', 'humidity', 'pressure']
    if 'pressure_change' in data.columns:
        feature_cols.append('pressure_change')
    
    available_features = [col for col in feature_cols if col in data.columns]
    
    # Prepare data
    X = data[available_features].fillna(data[available_features].mean())
    y = data['will_be_windy']
    
    # Train model
    model, accuracy, report = train_weather_model(X, y, "Wind")
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")
        
        # Wind probability distribution
        wind_probs = model.predict_proba(X)[:, 1]
        fig = px.histogram(wind_probs, nbins=20, title="💨 Wind Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Wind forecast
        st.write("**📅 Wind Forecast:**")
        forecast = make_weather_forecast(model, data, available_features, 'wind')
        
        for date, prob in forecast.items():
            wind_emoji = "💨" if prob > 0.6 else "🌬️" if prob > 0.3 else "🍃"
            st.write(f"{wind_emoji} {date}: {prob:.0%} chance of strong winds")

def create_sunny_predictor(data):
    """Create sunny weather prediction model"""
    st.subheader("☀️ Sunny Weather Prediction")
    
    feature_cols = ['temperature', 'humidity', 'pressure', 'heat_index']
    available_features = [col for col in feature_cols if col in data.columns]
    
    # Prepare data
    X = data[available_features].fillna(data[available_features].mean())
    y = data['will_be_sunny']
    
    # Train model
    model, accuracy, report = train_weather_model(X, y, "Sunny")
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")
        
        # Sunny days by temperature
        sunny_by_temp = data.groupby(pd.cut(data['temperature'], bins=10))['will_be_sunny'].mean()
        fig = px.bar(x=sunny_by_temp.index.astype(str), y=sunny_by_temp.values,
                    title="☀️ Sunny Probability by Temperature Range")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sunny forecast
        st.write("**📅 Sunshine Forecast:**")
        forecast = make_weather_forecast(model, data, available_features, 'sunny')
        
        for date, prob in forecast.items():
            sun_emoji = "☀️" if prob > 0.7 else "⛅" if prob > 0.4 else "☁️"
            st.write(f"{sun_emoji} {date}: {prob:.0%} chance of sunny weather")

def create_severe_weather_predictor(data):
    """Create severe weather prediction model"""
    st.subheader("🌪️ Severe Weather Prediction")
    
    st.warning("⚠️ **Disclaimer:** This is a demonstration model. For real severe weather warnings, always consult official meteorological services.")
    
    feature_cols = ['temperature', 'humidity', 'pressure', 'heat_index']
    if 'pressure_change' in data.columns:
        feature_cols.append('pressure_change')
    
    available_features = [col for col in feature_cols if col in data.columns]
    
    # Prepare data
    X = data[available_features].fillna(data[available_features].mean())
    y = data['severe_weather']
    
    # Train model
    model, accuracy, report = train_weather_model(X, y, "Severe Weather")
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")
        
        # Severe weather risk factors
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'factor': available_features,
                'risk_weight': model.feature_importances_
            }).sort_values('risk_weight', ascending=False)
            
            fig = px.bar(importance_df, x='risk_weight', y='factor',
                        title="⚠️ Severe Weather Risk Factors")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severe weather alerts
        st.write("**⚠️ Severe Weather Alerts:**")
        forecast = make_weather_forecast(model, data, available_features, 'severe')
        
        for date, prob in forecast.items():
            if prob > 0.3:
                alert_level = "🚨 HIGH" if prob > 0.6 else "⚠️ MODERATE" if prob > 0.3 else "✅ LOW"
                st.write(f"{alert_level} RISK {date}: {prob:.0%} severe weather probability")
            else:
                st.write(f"✅ {date}: Low risk ({prob:.0%})")

def train_weather_model(X, y, model_type):
    """Train a weather prediction model"""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return model, accuracy, report
        
    except Exception as e:
        st.error(f"❌ Error training {model_type} model: {str(e)}")
        return None, 0, {}

def make_weather_forecast(model, data, features, weather_type):
    """Generate weather forecast for next few days"""
    try:
        # Use the latest data point as base
        latest_data = data[features].iloc[-1:].fillna(data[features].mean())
        
        # Create variations for next 5 days
        forecast = {}
        base_date = datetime.now()
        
        for i in range(1, 6):
            # Add some realistic variation to the base data
            variation_factor = 1 + (np.random.normal(0, 0.1))
            forecast_data = latest_data.copy()
            
            # Apply variations
            for col in features:
                if col in ['temperature']:
                    forecast_data[col] *= variation_factor
                elif col in ['humidity']:
                    forecast_data[col] *= (1 + np.random.normal(0, 0.05))
                elif col in ['pressure']:
                    forecast_data[col] += np.random.normal(0, 5)
            
            # Make prediction
            prob = model.predict_proba(forecast_data)[0][1]
            
            forecast_date = (base_date + timedelta(days=i)).strftime("%m/%d")
            forecast[forecast_date] = prob
        
        return forecast
        
    except Exception as e:
        st.error(f"❌ Error generating forecast: {str(e)}")
        return {}

def create_interactive_rain_predictor(model, features):
    """Create interactive rain prediction interface"""
    st.subheader("🎯 Interactive Rain Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input controls
        user_inputs = {}
        user_inputs['temperature'] = st.slider("🌡️ Temperature (°C)", -10.0, 45.0, 20.0)
        user_inputs['humidity'] = st.slider("💧 Humidity (%)", 0.0, 100.0, 60.0)
        user_inputs['pressure'] = st.slider("🏔️ Pressure (hPa)", 980.0, 1040.0, 1013.0)
        
        if 'heat_index' in features:
            user_inputs['heat_index'] = calculate_heat_index(user_inputs['temperature'], user_inputs['humidity'])
        
        if 'dew_point' in features:
            user_inputs['dew_point'] = calculate_dew_point(user_inputs['temperature'], user_inputs['humidity'])
    
    with col2:
        # Make prediction
        if st.button("🔮 Predict Rain", type="primary"):
            try:
                # Prepare input data
                input_data = pd.DataFrame([user_inputs])
                input_data = input_data[features].fillna(0)
                
                # Make prediction
                rain_prob = model.predict_proba(input_data)[0][1]
                
                # Display result
                if rain_prob > 0.7:
                    st.success(f"☔ High chance of rain: {rain_prob:.0%}")
                    st.info("🌂 Recommendation: Take an umbrella!")
                elif rain_prob > 0.4:
                    st.warning(f"🌤️ Moderate chance of rain: {rain_prob:.0%}")
                    st.info("☁️ Recommendation: Keep an eye on the weather")
                else:
                    st.info(f"☀️ Low chance of rain: {rain_prob:.0%}")
                    st.success("😎 Recommendation: Great day to go outside!")
                
                # Show probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = rain_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Rain Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}") 