# Baby Names App

A comprehensive data pipeline and Streamlit application for exploring baby names popularity data from the Social Security Administration, featuring machine learning models for predicting future name popularity trends.

## Features

- **Data Pipeline**: Automatically downloads and processes baby names data from SSA.gov using Selenium
- **Machine Learning Models**: Multiple ML models (Linear Regression, Random Forest, XGBoost, LSTM) for predicting name popularity
- **Interactive Visualization**: Streamlit app with interactive charts and search functionality
- **Name Search**: Search for specific baby names and view their popularity metrics
- **Trend Analysis**: View popularity trends and rankings with predictive capabilities
- **Comparison Tools**: Compare multiple names side by side
- **Future Predictions**: Predict future popularity trends using trained ML models

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd baby-names-app
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline and start the app:
```bash
# Run the Jupyter notebook for data processing and ML model training
jupyter notebook data_pipeline.ipynb
```

This will:
1. Download and process the baby names data from SSA.gov using Selenium
2. Train multiple machine learning models for name popularity prediction
3. Save the best performing model for use in the Streamlit app

### Manual Steps

If you prefer to run steps manually:

1. **Run the data pipeline and ML training**:
```bash
jupyter notebook data_pipeline.ipynb
```
Execute all cells to:
- Download data from SSA.gov
- Process and clean the data
- Train ML models (Linear Regression, Random Forest, XGBoost, LSTM)
- Evaluate model performance and select the best model
- Save models for production use

2. **Start the Streamlit app**:
```bash
streamlit run app.py
```

## Data Pipeline

The `data_pipeline.ipynb` Jupyter notebook provides a comprehensive data processing and machine learning pipeline:

### Data Processing:
- Downloads the complete baby names dataset from SSA.gov (names.zip) using Selenium
- Extracts and processes data from all years (1880-2024) in the zip file
- Creates a comprehensive dataset with all name-sex combinations across all years
- Calculates popularity metrics (total count, popularity percentage, popularity rank)
- Saves the processed data to SQLite database (`data/names.db`)

### Machine Learning Pipeline:
- **Data Preparation**: Creates time series features with 5-year lookback windows
- **Feature Engineering**: Includes historical trends, volatility, and temporal patterns
- **Model Training**: Trains 4 different ML models:
  - Linear Regression (for interpretability)
  - Random Forest (for robust predictions)
  - XGBoost (for high performance)
  - LSTM (for complex time series patterns)
- **Model Evaluation**: Comprehensive performance assessment with RMSE, MAE, and R² metrics
- **Model Selection**: Automatically selects the best performing model
- **Model Persistence**: Saves trained models and scalers for production use

## Streamlit App

The `app.py` provides:
- **Name Search**: Enter a baby name to see its popularity metrics and trends
- **Top Names**: View the most popular names by gender and year
- **Trend Analysis**: Interactive charts showing popularity trends over time
- **Future Predictions**: ML-powered predictions for future name popularity
- **Comparison Tools**: Compare multiple names side by side
- **Interactive Charts**: Built with Plotly for rich visualizations

## Data Structure

The processed data includes:
- `name`: Baby name
- `sex`: M (Male) or F (Female)
- `total_count`: Number of occurrences in a specific year
- `year`: Year of birth (1880-2024)
- `popularity_percent`: Relative popularity as percentage of total births for that sex/year
- `popularity_rank`: Ranking within the sex/year (1 = most popular)

## Machine Learning Models

The pipeline trains and evaluates multiple models for name popularity prediction:

### Models Included:
1. **Linear Regression**: Fast, interpretable, good baseline
2. **Random Forest**: Robust, handles non-linear relationships
3. **XGBoost**: High performance, good for production
4. **LSTM**: Deep learning for complex time series patterns

### Features Used:
- Historical popularity percentages (5-year lookback)
- Historical total counts (5-year lookback)
- Historical popularity ranks (5-year lookback)
- Trend features (linear trends in popularity and counts)
- Volatility measures (standard deviation of recent popularity)
- Temporal features (normalized year)

### Model Selection:
The pipeline automatically selects the best performing model based on RMSE (Root Mean Square Error) and provides comprehensive performance analysis including MAE and R² scores.

## Requirements

- Python 3.7+
- pandas
- streamlit
- requests
- beautifulsoup4
- plotly
- lxml
- selenium
- scikit-learn
- xgboost
- tensorflow
- joblib
- jupyter
- matplotlib
- numpy

## File Structure

```
baby-names-app/
├── data_pipeline.ipynb          # Main data processing and ML training notebook
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── data/                        # Data directory
    ├── names.zip               # Downloaded SSA data
    ├── names.db                # SQLite database with processed data
    ├── best_model_*.pkl        # Trained ML models
    ├── scaler.pkl              # Feature scaler
    ├── feature_names.pkl       # Feature names for model interpretation
    └── yob*.txt                # Individual year data files
```

## Data Source

Data is sourced from the [Social Security Administration's baby names database](https://www.ssa.gov/oact/babynames/limits.html).

## Model Performance

The machine learning pipeline provides comprehensive model evaluation:
- **Performance Metrics**: RMSE, MAE, and R² scores for each model
- **Feature Importance**: Analysis of which features contribute most to predictions
- **Model Comparison**: Side-by-side comparison of all trained models
- **Visualization**: Charts showing prediction accuracy and feature importance

## License

This project is for educational purposes. Please respect the SSA's terms of use for their data.