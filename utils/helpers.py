import streamlit as st
import pandas as pd
import base64
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import plotly.io as pio

def apply_dark_mode():
    """Apply dark mode styling to the app"""
    st.markdown("""
    <style>
    .reportview-container {
        background: #0e1117;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: white;
        font-weight: bold;
    }
    .stSelectbox>div>div>select {
        background-color: #262730;
        color: white;
    }
    .stSlider>div>div>div>div {
        background: linear-gradient(to right, #ff6b6b, #4ecdc4);
    }
    </style>
    """, unsafe_allow_html=True)

def toggle_theme():
    """Toggle between light and dark theme"""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    if st.sidebar.button("🌓 Toggle Theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    if st.session_state.dark_mode:
        apply_dark_mode()

def generate_data_summary(df):
    """Generate comprehensive data summary"""
    if df is None:
        return None
    
    summary = {
        "Dataset Shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
        "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum(),
        "Numeric Columns": len(df.select_dtypes(include=['int64', 'float64']).columns),
        "Categorical Columns": len(df.select_dtypes(include=['object']).columns),
        "DateTime Columns": len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    return summary

def display_data_summary(df):
    """Display comprehensive data summary in an expandable section"""
    if df is None:
        return
    
    with st.expander("📋 Data Summary", expanded=False):
        summary = generate_data_summary(df)
        
        col1, col2 = st.columns(2)
        with col1:
            for key in list(summary.keys())[:4]:
                st.metric(key, summary[key])
        with col2:
            for key in list(summary.keys())[4:]:
                st.metric(key, summary[key])
        
        # Data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            st.subheader("Basic Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

def export_to_csv(df, filename="data_export.csv"):
    """Export dataframe to CSV with download link"""
    if df is None:
        return
    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def export_to_pdf(data_summary, filename="data_report.pdf"):
    """Export data summary to PDF report"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Data Analysis Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Summary table
        if data_summary:
            summary_data = [['Metric', 'Value']]
            for key, value in data_summary.items():
                summary_data.append([key, str(value)])
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
        
        doc.build(story)
        buffer.seek(0)
        
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📑 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")

def create_download_section(df):
    """Create a section with download options"""
    if df is None:
        return
    
    st.subheader("💾 Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Data to CSV"):
            export_to_csv(df, "exported_data.csv")
    
    with col2:
        if st.button("📋 Export Summary to PDF"):
            summary = generate_data_summary(df)
            export_to_pdf(summary, "data_summary_report.pdf")

def add_footer():
    """Add a footer to the app"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 12px;'>
        📊 Smart Data Predictor & Visualizer | Built with Streamlit & ❤️
        </div>
        """, 
        unsafe_allow_html=True
    ) 