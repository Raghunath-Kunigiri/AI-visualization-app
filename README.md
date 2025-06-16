# 📊 Smart Data Predictor & Visualizer

A comprehensive data analysis and machine learning web application built with Streamlit.

## 🚀 Features

- **📤 CSV Data Upload**: Easy drag-and-drop file upload
- **🎨 Interactive Visualizations**: Multiple chart types (Line, Scatter, Bar, Histogram, Box Plot)
- **🤖 Machine Learning Models**: Linear Regression, Random Forest, XGBoost
- **📊 Data Analysis**: Comprehensive data summaries and correlation matrices
- **🌓 Theme Toggle**: Switch between light and dark modes
- **💾 Export Options**: Download data as CSV and reports as PDF
- **📱 Responsive Design**: Works on desktop and mobile devices

## 📁 Project Structure

```
smart_data_app/
├── app.py                      # Main Streamlit application
├── components/
│   ├── __init__.py
│   ├── uploader.py             # CSV file upload and data handling
│   ├── visualizer.py           # Chart generation and data visualization
│   └── model_trainer.py        # ML model training and evaluation
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Theme toggle, export functions, utilities
├── assets/
│   └── style.css               # Custom CSS styling for dark mode
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🛠️ Installation

1. **Clone or download the project**:
   ```bash
   cd smart_data_app
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your CSV file** using the file uploader

4. **Explore your data**:
   - **Visualization Tab**: Create interactive charts
   - **Model Training Tab**: Train ML models and see predictions
   - **Data Analysis Tab**: View detailed data statistics
   - **Export Tab**: Download processed data and reports

## 📊 Supported Data Formats

- **CSV files** with numeric columns for analysis
- Files should contain at least 2 numeric columns for full functionality
- Missing values are handled automatically

## 🤖 Machine Learning Models

### Linear Regression
- Simple and interpretable
- Good for linear relationships
- Fast training

### Random Forest
- Handles non-linear relationships
- Feature importance available
- Robust to overfitting

### XGBoost
- High performance gradient boosting
- Excellent for complex patterns
- Advanced feature importance

## 📈 Visualization Types

- **Line Charts**: Time series and trend analysis
- **Scatter Plots**: Relationship exploration
- **Bar Charts**: Category comparisons
- **Histograms**: Distribution analysis
- **Box Plots**: Statistical summaries
- **Correlation Matrix**: Feature relationships

## 🎨 Customization

### Theme
- Toggle between light and dark modes using the sidebar button
- Custom CSS in `assets/style.css` for styling modifications

### Adding New Features
- **Components**: Add new functionality in the `components/` folder
- **Utilities**: Add helper functions in `utils/helpers.py`
- **Styling**: Modify `assets/style.css` for visual changes

## 📋 Dependencies

- **streamlit**: Web app framework
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning models
- **plotly**: Interactive visualizations
- **numpy**: Numerical computations
- **xgboost**: Gradient boosting (optional)
- **reportlab**: PDF generation

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CSS Not Loading**: Ensure the `assets/style.css` file exists in the correct location

3. **XGBoost Issues**: XGBoost is optional and will gracefully fallback if not available

4. **File Upload Problems**: Check that your CSV has numeric columns and proper formatting

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue with detailed error information

---

**Built with ❤️ using Streamlit, Pandas, and Plotly** 