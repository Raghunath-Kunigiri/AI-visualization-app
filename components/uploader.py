import streamlit as st
import pandas as pd

def upload_csv():
    """Handle CSV file upload and return dataframe"""
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Raw Data")
            st.write(df)
            return df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    return None

def get_numeric_columns(df):
    """Get numeric columns from dataframe"""
    if df is not None:
        return df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return []

def display_data_summary(df):
    """Display basic data summary"""
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum()) 