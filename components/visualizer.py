import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def create_chart_controls(numeric_columns):
    """Create sidebar controls for chart selection"""
    st.sidebar.header("🔧 Controls")
    
    if len(numeric_columns) < 2:
        st.sidebar.error("Need at least 2 numeric columns for visualization")
        return None, None, None
    
    x_axis = st.sidebar.selectbox("Select X-axis (Feature)", numeric_columns)
    y_axis = st.sidebar.selectbox("Select Y-axis (Target)", numeric_columns)
    chart_type = st.sidebar.radio("Select Chart Type", ['Line', 'Scatter', 'Bar', 'Histogram', 'Box Plot'])
    
    return x_axis, y_axis, chart_type

def generate_chart(df, x_axis, y_axis, chart_type):
    """Generate chart based on selected parameters"""
    if df is None or x_axis is None or y_axis is None:
        return None
    
    try:
        if chart_type == 'Line':
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{chart_type} Chart: {y_axis} vs {x_axis}")
        elif chart_type == 'Scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{chart_type} Chart: {y_axis} vs {x_axis}")
        elif chart_type == 'Bar':
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{chart_type} Chart: {y_axis} vs {x_axis}")
        elif chart_type == 'Histogram':
            fig = px.histogram(df, x=x_axis, title=f"{chart_type}: {x_axis}")
        elif chart_type == 'Box Plot':
            fig = px.box(df, y=y_axis, title=f"{chart_type}: {y_axis}")
        
        # Enhance chart appearance
        fig.update_layout(
            template="plotly_white",
            title_font_size=16,
            font=dict(size=12)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")
        return None

def display_correlation_matrix(df, numeric_columns):
    """Display correlation matrix heatmap"""
    if df is not None and len(numeric_columns) > 1:
        st.subheader("🔗 Correlation Matrix")
        corr_matrix = df[numeric_columns].corr()
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True) 