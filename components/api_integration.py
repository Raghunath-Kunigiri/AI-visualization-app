import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px

def create_api_data_section():
    """Create section for fetching data from APIs"""
    st.subheader("🌐 Real-time Data Integration")
    
    api_source = st.selectbox(
        "Select Data Source:",
        ["Custom API", "Sample Stock Data", "Weather Data", "Cryptocurrency"]
    )
    
    if api_source == "Sample Stock Data":
        return fetch_sample_stock_data()
    elif api_source == "Weather Data":
        return fetch_sample_weather_data()
    elif api_source == "Cryptocurrency":
        return fetch_crypto_data()
    elif api_source == "Custom API":
        return create_custom_api_section()
    
    return None

def fetch_sample_stock_data():
    """Generate sample stock-like data"""
    st.info("📊 Generating sample stock data...")
    
    # Generate sample data that mimics stock prices
    import numpy as np
    
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate stock price movement
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [100]
    
    for return_rate in returns[1:]:
        prices.append(prices[-1] * (1 + return_rate))
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(1000, 10000, len(dates)),
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    })
    
    st.success("✅ Sample stock data generated!")
    
    # Display sample chart
    fig = px.line(df, x='date', y='price', title='Sample Stock Price Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    return df

def fetch_sample_weather_data():
    """Fetch real weather data from OpenWeather API or generate sample data"""
    st.subheader("🌤️ Weather Data Options")
    
    # Option to choose between real API or sample data
    data_type = st.radio(
        "Choose data source:",
        ["🌐 Real OpenWeather API", "📊 Sample Weather Data"],
        horizontal=True
    )
    
    if data_type == "🌐 Real OpenWeather API":
        return fetch_real_weather_data()
    else:
        return fetch_sample_weather_data_only()

def fetch_real_weather_data():
    """Fetch real weather data from OpenWeather API"""
    st.info("🌤️ Connecting to OpenWeather API...")
    
    # API configuration
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input(
            "OpenWeather API Key:", 
            type="password",
            placeholder="Enter your API key from openweathermap.org",
            help="Get your free API key at https://openweathermap.org/api"
        )
        
        # Quick API key options for testing
        st.write("**Quick Test Options:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔑 Use Your Key"):
                st.session_state.api_key = "750fdd3735458509f960fdc6c8767f4a"
                st.rerun()
        with col_b:
            if st.button("🧪 Use Demo Key"):
                st.session_state.api_key = "b6907d289e10d714a6e88b30761fae22"
                st.rerun()
        with col_c:
            if st.button("🗑️ Clear Key"):
                st.session_state.api_key = ""
                st.rerun()
        
        # Use session state API key if available
        if 'api_key' in st.session_state and st.session_state.api_key:
            api_key = st.session_state.api_key
            st.info(f"🔑 Using API Key: `{api_key[:8]}...{api_key[-8:]}`")
        
        # Advanced: Custom API key testing
        with st.expander("🧪 Test Different API Keys", expanded=False):
            st.write("**Add and test multiple API keys:**")
            
            # Input for new API key
            new_key_name = st.text_input("Key Name:", placeholder="My Weather Key")
            new_api_key = st.text_input("API Key:", placeholder="Enter 32-character API key")
            
            if st.button("➕ Add Key") and new_key_name and new_api_key:
                if 'saved_keys' not in st.session_state:
                    st.session_state.saved_keys = {}
                st.session_state.saved_keys[new_key_name] = new_api_key
                st.success(f"✅ Added '{new_key_name}' to saved keys!")
                st.rerun()
            
            # Display saved keys
            if 'saved_keys' in st.session_state and st.session_state.saved_keys:
                st.write("**Saved API Keys:**")
                for key_name, key_value in st.session_state.saved_keys.items():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"🔑 **{key_name}**")
                    with col2:
                        st.code(f"{key_value[:8]}...{key_value[-8:]}")
                    with col3:
                        if st.button("Use", key=f"use_{key_name}"):
                            st.session_state.api_key = key_value
                            st.rerun()
            
            # API key validation tool
            st.divider()
            st.write("**🔍 API Key Validator:**")
            test_key = st.text_input("Test API Key:", placeholder="Paste any API key to validate")
            if st.button("🧪 Quick Validate") and test_key:
                try:
                    test_url = "http://api.openweathermap.org/data/2.5/weather"
                    test_params = {'q': 'London', 'appid': test_key, 'units': 'metric'}
                    test_response = requests.get(test_url, params=test_params, timeout=5)
                    
                    if test_response.status_code == 200:
                        st.success("✅ Valid API key!")
                        st.json({"status": "Valid", "plan": "Active", "city_tested": "London"})
                    elif test_response.status_code == 401:
                        st.error("❌ Invalid API key")
                    elif test_response.status_code == 429:
                        st.warning("⚠️ Valid key but rate limited")
                    else:
                        st.error(f"❌ Error: {test_response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Validation failed: {str(e)}")
    with col2:
        city = st.text_input("City:", value="London", placeholder="Enter city name")
    
    if st.button("🌍 Fetch Weather Data") and api_key and city:
        try:
            # Current weather endpoint
            current_url = f"http://api.openweathermap.org/data/2.5/weather"
            current_params = {
                'q': city,
                'appid': api_key,
                'units': 'metric'
            }
            
            # 5-day forecast endpoint
            forecast_url = f"http://api.openweathermap.org/data/2.5/forecast"
            forecast_params = {
                'q': city,
                'appid': api_key,
                'units': 'metric'
            }
            
            with st.spinner("🔄 Fetching current weather..."):
                current_response = requests.get(current_url, params=current_params, timeout=10)
                
                # Debug information
                st.write("🔍 **Debug Info:**")
                st.write(f"- API URL: {current_url}")
                st.write(f"- Status Code: {current_response.status_code}")
                st.write(f"- Response: {current_response.text[:200]}...")
                
                # Better error handling
                if current_response.status_code == 401:
                    st.error("❌ Invalid API key. Please check your OpenWeather API key.")
                    st.info(f"""
**Common API Key Issues:**
1. **New API Key**: Wait 10-60 minutes for activation
2. **Copy-Paste Error**: Make sure no extra spaces or characters
3. **Wrong Plan**: Free plan has limitations
4. **Account Issues**: Check your OpenWeather account status

**Your API Key (first/last 4 chars):** `{api_key[:4] if len(api_key) > 8 else api_key}...{api_key[-4:] if len(api_key) > 8 else ''}`
                    """)
                    return None
                elif current_response.status_code == 404:
                    st.error(f"❌ City '{city}' not found. Please check the city name.")
                    return None
                elif current_response.status_code == 429:
                    st.error("❌ API rate limit exceeded. Please wait a few minutes and try again.")
                    return None
                elif current_response.status_code != 200:
                    st.error(f"❌ API Error: {current_response.status_code} - {current_response.text}")
                    return None
                
                # Check if response is empty or not JSON
                if not current_response.text.strip():
                    st.error("❌ Empty response from API. Please try again.")
                    return None
                
                try:
                    current_data = current_response.json()
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON response: {str(e)}")
                    st.error(f"Response text: {current_response.text[:200]}...")
                    return None
            
            with st.spinner("🔄 Fetching 5-day forecast..."):
                forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
                
                if forecast_response.status_code != 200:
                    st.warning("⚠️ Could not fetch forecast data, using current weather only")
                    forecast_data = None
                else:
                    try:
                        forecast_data = forecast_response.json()
                    except json.JSONDecodeError:
                        st.warning("⚠️ Invalid forecast data, using current weather only")
                        forecast_data = None
            
            # Process current weather data
            weather_records = []
            current_time = datetime.now()
            
            # Add current weather
            weather_records.append({
                'datetime': current_time,
                'temperature': current_data['main']['temp'],
                'feels_like': current_data['main']['feels_like'],
                'humidity': current_data['main']['humidity'],
                'pressure': current_data['main']['pressure'],
                'wind_speed': current_data.get('wind', {}).get('speed', 0),
                'wind_direction': current_data.get('wind', {}).get('deg', 0),
                'weather': current_data['weather'][0]['description'],
                'city': current_data['name'],
                'country': current_data['sys']['country']
            })
            
            # Add forecast data if available
            if forecast_data and 'list' in forecast_data:
                for item in forecast_data['list']:
                    weather_records.append({
                        'datetime': datetime.fromtimestamp(item['dt']),
                        'temperature': item['main']['temp'],
                        'feels_like': item['main']['feels_like'],
                        'humidity': item['main']['humidity'],
                        'pressure': item['main']['pressure'],
                        'wind_speed': item.get('wind', {}).get('speed', 0),
                        'wind_direction': item.get('wind', {}).get('deg', 0),
                        'weather': item['weather'][0]['description'],
                        'city': city,
                        'country': current_data['sys']['country']
                    })
            
            df = pd.DataFrame(weather_records)
            
            st.success(f"✅ Successfully fetched weather data for {city}!")
            
            # Display current weather info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌡️ Temperature", f"{current_data['main']['temp']:.1f}°C")
            with col2:
                st.metric("💧 Humidity", f"{current_data['main']['humidity']}%")
            with col3:
                st.metric("🏔️ Pressure", f"{current_data['main']['pressure']} hPa")
            with col4:
                st.metric("💨 Wind Speed", f"{current_data.get('wind', {}).get('speed', 0)} m/s")
            
            st.info(f"🌤️ Current weather: {current_data['weather'][0]['description'].title()}")
            
            # Display temperature chart
            if len(df) > 1:
                fig = px.line(df, x='datetime', y='temperature', 
                             title=f'Temperature Forecast for {city}',
                             labels={'temperature': 'Temperature (°C)', 'datetime': 'Date & Time'})
                st.plotly_chart(fig, use_container_width=True)
            
            return df
            
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection error. Please check your internet connection.")
        except requests.exceptions.Timeout:
            st.error("❌ Request timeout. The API is taking too long to respond.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            
    elif api_key and city:
        st.info("👆 Click 'Fetch Weather Data' to get real weather information")
    elif not api_key:
        st.warning("⚠️ Please enter your OpenWeather API key")
        st.markdown("""
        **How to get an API key:**
        1. Go to [OpenWeather API Keys](https://home.openweathermap.org/api_keys)
        2. Create a free account or sign in
        3. Generate a new API key
        4. Copy and paste it above
        """)
    
    return None

def fetch_sample_weather_data_only():
    """Generate sample weather data"""
    st.info("🌤️ Generating sample weather data...")
    
    import numpy as np
    
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # Simulate seasonal temperature pattern
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    base_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    df = pd.DataFrame({
        'datetime': dates,
        'temperature': base_temp + np.random.normal(0, 3, len(dates)),
        'humidity': np.random.uniform(30, 90, len(dates)),
        'pressure': np.random.normal(1013, 10, len(dates)),
        'wind_speed': np.random.exponential(5, len(dates))
    })
    
    st.success("✅ Sample weather data generated!")
    
    # Display sample chart
    fig = px.line(df, x='datetime', y='temperature', title='Temperature Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    return df

def fetch_crypto_data():
    """Fetch sample cryptocurrency data"""
    st.info("₿ Generating sample cryptocurrency data...")
    
    import numpy as np
    
    # Generate realistic crypto price movements (more volatile)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    np.random.seed(42)
    
    returns = np.random.normal(0.002, 0.05, len(dates))  # Higher volatility
    prices = [30000]  # Starting price
    
    for return_rate in returns[1:]:
        prices.append(max(1000, prices[-1] * (1 + return_rate)))  # Minimum price floor
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.exponential(1000000, len(dates)),
        'market_cap': [p * 19000000 for p in prices],  # Simulate market cap
        'volatility': [abs(r) for r in returns]
    })
    
    st.success("✅ Sample cryptocurrency data generated!")
    
    # Display sample chart
    fig = px.line(df, x='date', y='price', title='Cryptocurrency Price Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    return df

def create_custom_api_section():
    """Create interface for custom API integration"""
    st.write("**Custom API Configuration:**")
    
    api_url = st.text_input("API URL:", placeholder="https://api.example.com/data")
    api_key = st.text_input("API Key (optional):", type="password")
    
    # Headers configuration
    st.write("**Headers:**")
    col1, col2 = st.columns(2)
    with col1:
        header_key = st.text_input("Header Key:", placeholder="Authorization")
    with col2:
        header_value = st.text_input("Header Value:", placeholder="Bearer token")
    
    # Parameters
    st.write("**Parameters:**")
    param_key = st.text_input("Parameter Key:", placeholder="limit")
    param_value = st.text_input("Parameter Value:", placeholder="100")
    
    if st.button("📡 Fetch Data"):
        if api_url:
            try:
                headers = {}
                if header_key and header_value:
                    headers[header_key] = header_value
                
                params = {}
                if param_key and param_value:
                    params[param_key] = param_value
                
                response = requests.get(api_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Try to convert to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                    else:
                        df = pd.DataFrame([data])
                else:
                    st.error("Unable to parse API response into DataFrame")
                    return None
                
                st.success(f"✅ Successfully fetched {len(df)} records!")
                st.dataframe(df.head())
                return df
                
            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {str(e)}")
            except Exception as e:
                st.error(f"Data processing error: {str(e)}")
        else:
            st.warning("Please enter an API URL")
    
    return None

def create_real_time_dashboard(df):
    """Create a real-time style dashboard"""
    if df is None:
        return
    
    st.subheader("📈 Real-time Dashboard")
    
    # Refresh option
    col1, col2 = st.columns([3, 1])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds")
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    if auto_refresh:
        st.info("⏱️ Dashboard will auto-refresh in 30 seconds")
        # This would require additional setup for actual auto-refresh
    
    # Current time
    st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Key metrics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.subheader("📊 Key Metrics")
        
        metrics_cols = st.columns(min(4, len(numeric_cols)))
        for i, col in enumerate(numeric_cols[:4]):
            with metrics_cols[i]:
                current_value = df[col].iloc[-1] if len(df) > 0 else 0
                previous_value = df[col].iloc[-2] if len(df) > 1 else current_value
                change = current_value - previous_value
                change_pct = (change / previous_value * 100) if previous_value != 0 else 0
                
                st.metric(
                    label=col.title(),
                    value=f"{current_value:.2f}",
                    delta=f"{change_pct:.1f}%"
                )
    
    # Time series charts
    if 'date' in df.columns or any('time' in col.lower() for col in df.columns):
        date_col = 'date' if 'date' in df.columns else next(col for col in df.columns if 'time' in col.lower())
        
        for col in numeric_cols[:3]:  # Show top 3 numeric columns
            fig = px.line(df, x=date_col, y=col, title=f"{col.title()} Over Time")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def create_data_streaming_simulation():
    """Simulate data streaming for demo purposes"""
    st.subheader("🌊 Data Streaming Simulation")
    
    if st.button("🎯 Start Streaming Simulation"):
        placeholder = st.empty()
        
        for i in range(10):
            # Generate new data point
            import numpy as np
            timestamp = datetime.now() - timedelta(seconds=10-i)
            value = 100 + np.random.normal(0, 5)
            
            # Create mini dataframe
            mini_df = pd.DataFrame({
                'timestamp': [timestamp],
                'value': [value],
                'status': ['normal' if abs(value - 100) < 10 else 'alert']
            })
            
            with placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Value", f"{value:.2f}")
                with col2:
                    st.metric("Status", mini_df['status'].iloc[0])
                
                # Simple line chart
                fig = px.line(mini_df, x='timestamp', y='value', 
                             title="Live Data Stream")
                st.plotly_chart(fig, use_container_width=True)
            
            # Simulate delay
            import time
            time.sleep(1)
        
        st.success("🎉 Streaming simulation completed!")

def create_webhook_simulator():
    """Create a webhook simulation interface"""
    st.subheader("🔗 Webhook Integration")
    
    st.write("**Simulate incoming webhook data:**")
    
    webhook_data = st.text_area(
        "JSON Payload:",
        value='{"sensor_id": "temp_01", "value": 23.5, "timestamp": "2024-01-01T12:00:00Z"}',
        height=100
    )
    
    if st.button("📡 Process Webhook Data"):
        try:
            data = json.loads(webhook_data)
            df = pd.DataFrame([data])
            
            st.success("✅ Webhook data processed successfully!")
            st.dataframe(df)
            
            return df
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON format")
    
    return None 