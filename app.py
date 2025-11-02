"""
Streamlit App for Baby Names Visualization
Allows users to search for baby names and view their popularity trends
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sqlite3
import logging
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from groq import Groq
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Baby Names Popularity Explorer",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_top_names_data(sex, top_n, year):
    """Load and aggregate the baby names data for top names"""
    conn = sqlite3.connect("data/names.db")
    df = pd.read_sql_query(f"SELECT * FROM names WHERE sex = '{sex}' AND year = {year} ORDER BY total_count DESC LIMIT {top_n}", conn)
    conn.close()
    return df

@st.cache_data
def load_name_search_data(name, sex):
    """Load and aggregate the baby names data for name search"""
    conn = sqlite3.connect("data/names.db")
    df = pd.read_sql_query(f"SELECT * FROM names WHERE LOWER(name) = LOWER('{name}') AND sex = '{sex}'", conn)
    conn.close()
    return df

def create_popularity_trend_chart(raw_df, name, sex, type='percent'):
    """Create a trend chart showing popularity over years with ARIMA projection"""
    # Sort by year
    raw_df = raw_df.sort_values('year')
    
    # Create line chart
    fig = go.Figure()
    
    if type == 'percent':
        y = raw_df['popularity_percent']
    else:
        y = raw_df['popularity_rank']

    # Add historical data
    fig.add_trace(go.Scatter(
        x=raw_df['year'],
        y=y,
        mode='lines+markers',
        name=f'{name} Historical Data',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add ARIMA projection if we have enough data
    if len(raw_df) >= 10:  # Need at least 10 years of data
        try:
            # Prepare time series data
            series = y.reset_index(drop=True)
            
            # Fit ARIMA model
            model, params = fit_arima_model(series)
            
            if model is not None:
                # Generate 5-year forecast
                forecast_mean, confidence_intervals = forecast_arima_with_uncertainty(model, steps=5)
                
                if forecast_mean is not None and confidence_intervals is not None:
                    # Create future years
                    last_year = raw_df['year'].max()
                    future_years = list(range(last_year + 1, last_year + 6))
                    
                    # Add forecast line
                    fig.add_trace(go.Scatter(
                        x=future_years,
                        y=forecast_mean,
                        mode='lines+markers',
                        name=f'{name} ARIMA Forecast',
                        line=dict(color='#ff7f0e', width=3, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Add confidence interval
                    fig.add_trace(go.Scatter(
                        x=future_years + future_years[::-1],
                        y=list(confidence_intervals.iloc[:, 1]) + list(confidence_intervals.iloc[:, 0])[::-1],
                        fill='tonexty',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval',
                        showlegend=True
                    ))
                    
                    # Add vertical line to separate historical and forecast
                    fig.add_vline(
                        x=last_year + 0.5,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text="Forecast Start"
                    )
                    
        except Exception as e:
            logger.warning(f"ARIMA projection failed for {name}: {e}")
    
    fig.update_layout(
        title=f"Popularity {type.capitalize()} Trend for {name} ({sex}) with 5-Year ARIMA Forecast",
        xaxis_title="Year",
        yaxis_title=f"Popularity {type.capitalize()}",
        height=500,
        hovermode='x unified'
    )

    return fig

def create_combined_trend_chart(raw_df, name, sex, type='percent'):
    """Create a combined trend chart showing historical data, ARIMA projection, and validation results"""
    # Sort by year
    raw_df = raw_df.sort_values('year')
    
    # Create line chart
    fig = go.Figure()
    
    if type == 'percent':
        y = raw_df['popularity_percent']
    else:
        y = raw_df['popularity_rank']

    # Add historical data
    fig.add_trace(go.Scatter(
        x=raw_df['year'],
        y=y,
        mode='lines+markers',
        name=f'{name} Historical Data',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add robust ARIMA projection and validation if we have enough data
    if len(raw_df) >= 10:  # Need at least 10 years of data
        try:
            # Prepare time series data
            series = y.reset_index(drop=True)
            
            # Show loading bar for ARIMA model fitting
            with st.spinner("🔮 Fitting ARIMA model and calculating projections..."):
                # Fit robust ARIMA model with progress bar
                arima_result = fit_robust_arima_model(series, show_progress=True)
                
                if arima_result is not None:
                    # Generate 5-year forecast with multiple confidence intervals
                    forecast_mean, confidence_intervals, arima_info = forecast_arima_with_uncertainty(arima_result, steps=5)
                    
                    if forecast_mean is not None and confidence_intervals is not None:
                        # Create future years
                        last_year = raw_df['year'].max()
                        future_years = list(range(last_year + 1, last_year + 6))
                        
                        # Add forecast line
                        fig.add_trace(go.Scatter(
                            x=future_years,
                            y=forecast_mean,
                            mode='lines+markers',
                            name=f'{name} ARIMA Forecast',
                            line=dict(color='#ff7f0e', width=3, dash='dash'),
                            marker=dict(size=8)
                        ))
                        
                        # Add 80% confidence interval (lighter)
                        if 0.8 in confidence_intervals:
                            ci_80 = confidence_intervals[0.8]
                            fig.add_trace(go.Scatter(
                                x=future_years + future_years[::-1],
                                y=list(ci_80['upper']) + list(ci_80['lower'])[::-1],
                                fill='tonexty',
                                fillcolor='rgba(255, 127, 14, 0.1)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='80% Confidence Interval',
                                showlegend=True
                            ))
                        
                        # Add 95% confidence interval (darker)
                        if 0.95 in confidence_intervals:
                            ci_95 = confidence_intervals[0.95]
                            fig.add_trace(go.Scatter(
                                x=future_years + future_years[::-1],
                                y=list(ci_95['upper']) + list(ci_95['lower'])[::-1],
                                fill='tonexty',
                                fillcolor='rgba(255, 127, 14, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% Confidence Interval',
                                showlegend=True
                            ))
                        
                        # Add validation results if we have enough data
                        if len(raw_df) >= 15:
                            validation_results, forecast_test, actual_test, validation_arima = validate_arima_model_robust(series, test_years=5, show_progress=False)
                            
                            if validation_results is not None and forecast_test is not None and actual_test is not None:
                                # Get the years for the test period
                                test_years = raw_df['year'].tail(5).tolist()
                                
                                # Add validation actual data (highlighted)
                                fig.add_trace(go.Scatter(
                                    x=test_years,
                                    y=actual_test,
                                    mode='lines+markers',
                                    name='Validation Actual (Last 5 Years)',
                                    line=dict(color='#2ca02c', width=4),
                                    marker=dict(size=10, symbol='diamond')
                                ))
                                
                                # Add validation predictions
                                fig.add_trace(go.Scatter(
                                    x=test_years,
                                    y=forecast_test,
                                    mode='lines+markers',
                                    name='Validation Predictions (Last 5 Years)',
                                    line=dict(color='#d62728', width=3, dash='dot'),
                                    marker=dict(size=8, symbol='square')
                                ))
                        
                        # Add vertical line to separate historical and forecast
                        fig.add_vline(
                            x=last_year + 0.5,
                            line_dash="dot",
                            line_color="gray",
                            annotation_text="Forecast Start"
                        )
                        
                        # Store ARIMA result for later use in validation metrics
                        st.session_state[f'{name}_{sex}_arima'] = arima_result
                    
        except Exception as e:
            logger.warning(f"ARIMA projection failed for {name}: {e}")
    
    fig.update_layout(
        title=f"Popularity {type.capitalize()} Trend for {name} ({sex}) with ARIMA Forecast & Validation",
        xaxis_title="Year",
        yaxis_title=f"Popularity {type.capitalize()}",
        height=600,
        hovermode='x unified'
    )
    
    return fig

def create_current_ranking_chart(raw_df, name):
    """Create a chart showing current ranking for the most recent year"""
    # Get the most recent year
    most_recent_year = raw_df['year'].max()
    previous_year = most_recent_year - 1
    
    # Filter data for the most recent year and specific sex
    name_data = raw_df[(raw_df['year'] == most_recent_year)]
    previous_name_data = raw_df[(raw_df['year'] == previous_year)]

    if name_data.empty:
        return None, None, None, None
    
    current_ranking = name_data['popularity_rank'].iloc[0]
    current_popularity = name_data['popularity_percent'].iloc[0]*100
    previous_ranking = previous_name_data['popularity_rank'].iloc[0]
    previous_popularity = previous_name_data['popularity_percent'].iloc[0]*100
    
    # Create ranking indicator
    last_digit = current_ranking % 10
    ranking_fig = go.Figure()
    ranking_fig.add_trace(go.Indicator(
        mode="number+delta",
        value=current_ranking,
        title={"text": f"Current Popularity Ranking ({most_recent_year})"},
        number={'suffix': "st" if last_digit == 1 else "nd" if last_digit == 2 else "rd" if last_digit == 3 else "th"},
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': previous_ranking, 'relative': False, 'valueformat': '.0f'}
    ))
    ranking_fig.update_layout(height=300)
    
    # Create popularity percent indicator
    popularity_fig = go.Figure()
    popularity_fig.add_trace(go.Indicator(
        mode="number+delta",
        value=current_popularity,
        title={"text": f"Percent of Babies with the Name {name} ({most_recent_year})"},
        number={'suffix': "%", 'valueformat': '.4f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': previous_popularity, 'relative': False, 'valueformat': '.2f'}
    ))
    popularity_fig.update_layout(height=300)
    
    return ranking_fig, popularity_fig, current_ranking, current_popularity

def preprocess_series(series):
    """Preprocess time series data following best practices"""
    # Remove any NaN values
    series = series.dropna()
    
    # Apply log transformation if series has positive values and high variance
    # Best practice: Use log transformation for multiplicative trends
    if (series > 0).all() and series.std() / series.mean() > 0.5:
        series = np.log1p(series)  # log1p handles zeros better than log
        return series, True
    else:
        return series, False

def inverse_transform(series, log_applied):
    """Inverse transform the series if log transformation was applied"""
    if log_applied:
        return np.expm1(series)  # Inverse of log1p
    return series

def check_stationarity(series):
    """Comprehensive stationarity testing following best practices"""
    series_clean = series.dropna()
    
    if len(series_clean) < 4:
        return False, 1.0, 1.0
    
    # ADF test (null hypothesis: series has unit root)
    adf_result = adfuller(series_clean, autolag='AIC')
    adf_pvalue = adf_result[1]
    
    # KPSS test (null hypothesis: series is stationary)
    try:
        kpss_result = kpss(series_clean, regression='c', nlags='auto')
        kpss_pvalue = kpss_result[1]
    except:
        kpss_pvalue = 1.0
    
    # Series is stationary if ADF rejects unit root (p < 0.05) 
    # AND KPSS fails to reject stationarity (p > 0.05)
    is_stationary = adf_pvalue < 0.05 and kpss_pvalue > 0.05
    
    return is_stationary, adf_pvalue, kpss_pvalue

def find_optimal_differencing(series, max_d=2):
    """Find optimal differencing order using systematic approach"""
    current_series = series.copy()
    d = 0
    
    for i in range(max_d + 1):
        is_stationary, adf_p, kpss_p = check_stationarity(current_series)
        if is_stationary:
            return d, current_series
        else:
            if i < max_d:
                current_series = current_series.diff().dropna()
                d += 1
    
    return d, current_series

def find_best_arima_params(series, max_p=4, max_d=2, max_q=4, show_progress=False):
    """Systematic ARIMA parameter selection using information criteria"""
    best_aic = float('inf')
    best_aicc = float('inf')
    best_params = None
    best_model = None
    
    # Ensure we have enough data
    if len(series) < 10:
        return (1, 1, 1), None
    
    # Find optimal differencing first
    optimal_d, stationary_series = find_optimal_differencing(series, max_d)
    
    # Search around optimal differencing with expanded range
    d_range = range(max(0, optimal_d-1), min(max_d+1, optimal_d+2))
    
    # Calculate total number of combinations for progress bar
    total_combinations = (max_p + 1) * len(d_range) * (max_q + 1)
    current_combination = 0
    
    # Create progress bar if requested
    progress_bar = None
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Systematic grid search
    for p in range(max_p + 1):
        for d in d_range:
            for q in range(max_q + 1):
                current_combination += 1
                
                # Update progress bar
                if progress_bar is not None:
                    progress = current_combination / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing ARIMA({p},{d},{q}) - {current_combination}/{total_combinations}")
                
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    # Use AICc for small samples (more conservative)
                    n = len(series)
                    k = p + q + 1  # Number of parameters
                    aicc = fitted_model.aic + (2 * k * (k + 1)) / (n - k - 1)
                    
                    # Prefer AICc for model selection
                    if aicc < best_aicc:
                        best_aicc = aicc
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except Exception as e:
                    continue
    
    # Clear progress bar
    if progress_bar is not None:
        progress_bar.empty()
        status_text.empty()
    
    return best_params if best_params is not None else (1, 1, 1), best_model

def check_residual_diagnostics(model):
    """Comprehensive residual diagnostics following best practices"""
    residuals = model.resid
    
    diagnostics = {}
    
    # 1. Ljung-Box test for residual autocorrelation
    try:
        lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=False)
        diagnostics['ljung_box'] = {
            'statistic': lb_stat[-1] if len(lb_stat) > 0 else 0,
            'p_value': lb_pvalue[-1] if len(lb_pvalue) > 0 else 1,
            'is_white_noise': lb_pvalue[-1] > 0.05 if len(lb_pvalue) > 0 else True
        }
    except:
        diagnostics['ljung_box'] = {'is_white_noise': True, 'p_value': 1.0}
    
    # 2. Normality test (Jarque-Bera)
    try:
        jb_stat, jb_pvalue = jarque_bera(residuals)
        diagnostics['normality'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'is_normal': jb_pvalue > 0.05
        }
    except:
        diagnostics['normality'] = {'is_normal': True, 'p_value': 1.0}
    
    # 3. Heteroscedasticity test (ARCH)
    try:
        arch_stat, arch_pvalue = het_arch(residuals)
        diagnostics['heteroscedasticity'] = {
            'statistic': arch_stat,
            'p_value': arch_pvalue,
            'is_homoscedastic': arch_pvalue > 0.05
        }
    except:
        diagnostics['heteroscedasticity'] = {'is_homoscedastic': True, 'p_value': 1.0}
    
    # 4. Overall model quality
    diagnostics['overall_quality'] = (
        diagnostics['ljung_box']['is_white_noise'] and
        diagnostics['normality']['is_normal'] and
        diagnostics['heteroscedasticity']['is_homoscedastic']
    )
    
    return diagnostics

def fit_robust_arima_model(series, show_progress=False):
    """Fit a robust ARIMA model following best practices"""
    # Preprocess the series
    processed_series, log_applied = preprocess_series(series)
    
    try:
        # Find best parameters using systematic approach
        best_params, best_model = find_best_arima_params(processed_series, show_progress=show_progress)
        
        if best_model is None:
            # Fallback: fit model with best parameters
            model = ARIMA(processed_series, order=best_params)
            fitted_model = model.fit()
        else:
            fitted_model = best_model
        
        # Perform residual diagnostics
        diagnostics = check_residual_diagnostics(fitted_model)
        
        # Check stationarity of original series
        is_stationary, adf_p, kpss_p = check_stationarity(processed_series)
        
        return {
            'model': fitted_model,
            'params': best_params,
            'log_applied': log_applied,
            'diagnostics': diagnostics,
            'stationarity': {
                'is_stationary': is_stationary,
                'adf_pvalue': adf_p,
                'kpss_pvalue': kpss_p
            },
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
        
    except Exception as e:
        logger.warning(f"ARIMA model fitting failed: {e}")
        return None

def forecast_arima_with_uncertainty(arima_result, steps=5, confidence_levels=[0.8, 0.95]):
    """Generate ARIMA forecast with multiple confidence intervals"""
    if arima_result is None:
        return None, None, None
    
    model = arima_result['model']
    log_applied = arima_result['log_applied']
    
    try:
        # Generate forecast
        forecast_result = model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        
        # Get confidence intervals for different levels
        confidence_intervals = {}
        for level in confidence_levels:
            conf_int = forecast_result.conf_int(alpha=1-level)
            confidence_intervals[level] = {
                'lower': conf_int.iloc[:, 0],
                'upper': conf_int.iloc[:, 1]
            }
        
        # Inverse transform if log was applied
        forecast_mean = inverse_transform(forecast_mean, log_applied)
        for level in confidence_intervals:
            confidence_intervals[level]['lower'] = inverse_transform(confidence_intervals[level]['lower'], log_applied)
            confidence_intervals[level]['upper'] = inverse_transform(confidence_intervals[level]['upper'], log_applied)
        
        return forecast_mean, confidence_intervals, arima_result
        
    except Exception as e:
        logger.warning(f"ARIMA forecasting failed: {e}")
        return None, None, None

def fit_arima_model(series, params=None):
    """Fit ARIMA model to the time series with preprocessing"""
    # Preprocess the series
    processed_series, log_applied = preprocess_series(series)
    
    if params is None:
        params = find_best_arima_params(processed_series)
    
    try:
        model = ARIMA(processed_series, order=params)
        fitted_model = model.fit()
        return fitted_model, params, log_applied
    except Exception as e:
        logger.warning(f"ARIMA model fitting failed: {e}")
        return None, None, None


def validate_arima_model_robust(series, test_years=5, show_progress=False):
    """Validate robust ARIMA model using the last test_years as test data"""
    if len(series) < test_years + 10:  # Need at least 10 years for training
        return None, None, None, None
    
    # Split data
    train_data = series[:-test_years]
    test_data = series[-test_years:]
    
    # Fit robust ARIMA model on training data
    arima_result = fit_robust_arima_model(train_data, show_progress=show_progress)
    if arima_result is None:
        return None, None, None, None
    
    # Generate forecasts for test period
    forecast_mean, confidence_intervals, arima_info = forecast_arima_with_uncertainty(arima_result, steps=test_years)
    if forecast_mean is None:
        return None, None, None, None
    
    # Calculate metrics
    mae = mean_absolute_error(test_data, forecast_mean)
    mse = mean_squared_error(test_data, forecast_mean)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error) - handle division by zero
    mape = np.mean(np.abs((test_data - forecast_mean) / np.maximum(test_data, 1e-8))) * 100
    
    # Calculate additional metrics
    mape_median = np.median(np.abs((test_data - forecast_mean) / np.maximum(test_data, 1e-8))) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'mape_median': mape_median,
        'arima_info': arima_info,
        'model_quality': arima_info['diagnostics']['overall_quality']
    }, forecast_mean, test_data, arima_result

def validate_arima_model(series, test_years=5):
    """Validate ARIMA model using the last test_years as test data (fallback)"""
    if len(series) < test_years + 10:  # Need at least 10 years for training
        return None, None, None
    
    # Split data
    train_data = series[:-test_years]
    test_data = series[-test_years:]
    
    # Fit model on training data
    model, params, log_applied = fit_arima_model(train_data)
    if model is None:
        return None, None, None
    
    # Generate forecasts for test period using the old method
    try:
        forecast = model.get_forecast(steps=test_years)
        forecast_mean = forecast.predicted_mean
        # Inverse transform if log was applied
        forecast_mean = inverse_transform(forecast_mean, log_applied)
    except Exception as e:
        logger.warning(f"ARIMA forecasting failed: {e}")
        forecast_mean = None
    if forecast_mean is None:
        return None, None, None
    
    # Calculate metrics
    mae = mean_absolute_error(test_data, forecast_mean)
    mse = mean_squared_error(test_data, forecast_mean)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error) - handle division by zero
    mape = np.mean(np.abs((test_data - forecast_mean) / np.maximum(test_data, 1e-8))) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'params': params
    }, forecast_mean, test_data

def main_page():
    """A description of the app, it's purpose and how to use it"""
    st.title("👶 Baby Names Popularity Explorer")
    st.markdown("Explore the popularity of baby names from the Social Security Administration dataset")
    st.markdown("This app allows you to ask the AI Chatbot any questions about the baby names database in natural language")
    st.markdown("Search for specific baby names and view their popularity metrics and projections")
    st.markdown("You can also view the top names by year and gender")
    st.markdown("The data is sourced from the [Social Security Administration's baby names database](https://www.ssa.gov/oact/babynames/limits.html)")
    st.markdown("The data is updated yearly")
    
def top_names_page():
    """Main page showing top names and general statistics"""
    st.title("👶 Top Names")
    st.markdown("Explore the top names from the Social Security Administration dataset by year and gender")
    
    # Create columns for filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sex filter
        sex = st.selectbox("Select Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
    
    with col2:
        # Top names filter
        top_n = st.number_input(
            "Number of top names to show", 
            min_value=10, 
            max_value=100, 
            value=20, 
            step=5
        )

    with col3:
        # Year filter
        year = st.number_input(
            "Year", 
            min_value=1880, 
            max_value=2024, 
            value=2024, 
            step=1
        )

    # Load data
    top_names_df = load_top_names_data(sex, top_n, year)
    if top_names_df is None:
        st.stop()
    
    # Top names section
    st.header(f"Top {top_n} {"Male" if sex == "M" else "Female"} Names")
    
    # Get top names
    top_names = top_names_df[top_names_df['sex'] == sex].head(top_n)
    
    # Create top names chart
    fig = px.bar(
        top_names,
        x='name',
        y='total_count',
        title=f"Top {top_n} {"Male" if sex == "M" else "Female"} Names by Total Count",
        labels={'total_count': 'Total Count', 'name': 'Name'},
        color='total_count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top names table
    st.subheader("Top Names Data")
    st.dataframe(top_names, width='stretch')

def search_page():
    """Name search page"""
    st.title("🔍 Name Search")
    st.markdown("Search for specific baby names and view their popularity metrics and projections")
    
    # Search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_name = st.text_input("Enter a baby name:", placeholder="e.g., Emma, Liam", key="search_input")
    
    with col2:
        sex = st.selectbox("Select Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female", key="search_sex")

    # Search functionality
    if search_name: 
        # Load data
        search_df = load_name_search_data(search_name, sex)
        if search_df is None:
            st.stop()      
        
        if not search_df.empty:            
            # Top row: Current ranking and detailed info
            col1, col2 = st.columns(2)
            
            with col1:
                ranking_result = create_current_ranking_chart(search_df, search_name)
                if ranking_result:
                    if ranking_result[0] is not None:
                        ranking_fig, popularity_fig, current_ranking, current_popularity = ranking_result
                        st.plotly_chart(ranking_fig, use_container_width=True)
                        st.plotly_chart(popularity_fig, use_container_width=True)
            
            with col2:
                # Show year-by-year data
                if not search_df.empty:
                    search_df = search_df.sort_values('year', ascending=False)
                    st.subheader("Year-by-Year Data")
                    # Create a copy and convert to percentage
                    display_data = search_df[['year', 'total_count', 'popularity_percent', 'popularity_rank']].copy()
                    display_data['popularity_percent'] = display_data['popularity_percent'] * 100
                    st.dataframe(
                        display_data[['year', 'total_count', 'popularity_percent', 'popularity_rank']], 
                        width='stretch', height=600, hide_index=True,
                        column_config={
                            "year": st.column_config.NumberColumn(
                                "Year",
                                help="The year the data was recorded",
                                format="%d"
                            ),
                            "total_count": st.column_config.NumberColumn(
                                "Total Count",
                                help="Total number of babies with this name",
                                format="%d"
                            ),
                            "popularity_percent": st.column_config.NumberColumn(
                                "Popularity %",
                                help="Percentage of babies with this name",
                                format="%.6f%%"
                            ),
                            "popularity_rank": st.column_config.NumberColumn(
                                "Rank",
                                help="Ranking among all names",
                                format="%d"
                            )
                        })
            
            # Combined trend chart with ARIMA projection and validation
            combined_chart = create_combined_trend_chart(search_df, search_name, sex, type='percent')
            if combined_chart:
                st.plotly_chart(combined_chart, use_container_width=True)
            
            # Robust ARIMA Model Performance Metrics
            if len(search_df) >= 10:  # Need at least 10 years for meaningful validation
                st.subheader("📊 Robust ARIMA Model Performance")
                st.markdown("Model validation using the 5 most recent years as test data:")
                
                # Prepare data for validation (always use percent for consistency)
                validation_series = search_df.sort_values('year')['popularity_percent'].reset_index(drop=True)
                
                # Show loading bar for validation
                with st.spinner("📊 Calculating model validation metrics..."):
                    # Validate robust ARIMA model with progress bar
                    validation_results, forecast_test, actual_test, validation_arima = validate_arima_model_robust(validation_series, test_years=5, show_progress=True)
                
                if validation_results is not None:
                    # Display main metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean Absolute Error", f"{validation_results['mae']:.6f}")
                    
                    with col2:
                        st.metric("Root Mean Square Error", f"{validation_results['rmse']:.6f}")
                    
                    with col3:
                        st.metric("Mean Absolute % Error", f"{validation_results['mape']:.2f}%")
                    
                    with col4:
                        # Show model quality indicator
                        quality_icon = "✅" if validation_results['model_quality'] else "⚠️"
                        st.metric("Model Quality", f"{quality_icon} {'Good' if validation_results['model_quality'] else 'Needs Review'}")
                    
                    # Show detailed model information
                    st.subheader("🔍 Model Diagnostics & Details")
                    
                    # Create tabs for different information
                    tab1, tab2, tab3 = st.tabs(["Model Parameters", "Residual Diagnostics", "Stationarity Tests"])
                    
                    with tab1:
                        arima_info = validation_results['arima_info']
                        st.markdown("**ARIMA Model Information:**")
                        st.markdown(f"- **Order (p,d,q):** {arima_info['params']}")
                        st.markdown(f"- **AIC:** {arima_info['aic']:.2f}")
                        st.markdown(f"- **BIC:** {arima_info['bic']:.2f}")
                        st.markdown(f"- **Log Transform Applied:** {'Yes' if arima_info['log_applied'] else 'No'}")
                        
                        # Additional metrics
                        st.markdown("**Additional Performance Metrics:**")
                        st.markdown(f"- **Median MAPE:** {validation_results['mape_median']:.2f}%")
                    
                    with tab2:
                        diagnostics = arima_info['diagnostics']
                        st.markdown("**Residual Diagnostics:**")
                        
                        # Ljung-Box test
                        lb = diagnostics['ljung_box']
                        lb_status = "✅ Pass" if lb['is_white_noise'] else "❌ Fail"
                        st.markdown(f"- **Ljung-Box Test (White Noise):** {lb_status} (p={lb['p_value']:.4f})")
                        
                        # Normality test
                        norm = diagnostics['normality']
                        norm_status = "✅ Pass" if norm['is_normal'] else "❌ Fail"
                        st.markdown(f"- **Normality Test:** {norm_status} (p={norm['p_value']:.4f})")
                        
                        # Heteroscedasticity test
                        het = diagnostics['heteroscedasticity']
                        het_status = "✅ Pass" if het['is_homoscedastic'] else "❌ Fail"
                        st.markdown(f"- **Heteroscedasticity Test:** {het_status} (p={het['p_value']:.4f})")
                        
                        # Overall quality
                        overall_status = "✅ Good" if diagnostics['overall_quality'] else "⚠️ Needs Review"
                        st.markdown(f"- **Overall Model Quality:** {overall_status}")
                    
                    with tab3:
                        stationarity = arima_info['stationarity']
                        st.markdown("**Stationarity Tests:**")
                        
                        # ADF test
                        adf_status = "✅ Stationary" if stationarity['adf_pvalue'] < 0.05 else "❌ Non-stationary"
                        st.markdown(f"- **ADF Test:** {adf_status} (p={stationarity['adf_pvalue']:.4f})")
                        
                        # KPSS test
                        kpss_status = "✅ Stationary" if stationarity['kpss_pvalue'] > 0.05 else "❌ Non-stationary"
                        st.markdown(f"- **KPSS Test:** {kpss_status} (p={stationarity['kpss_pvalue']:.4f})")
                        
                        # Combined result
                        combined_status = "✅ Stationary" if stationarity['is_stationary'] else "❌ Non-stationary"
                        st.markdown(f"- **Combined Result:** {combined_status}")
                
                else:
                    st.warning("Insufficient data for model validation. Need at least 10 years of data.")
            else:
                st.info("Model validation requires at least 10 years of data. Current data spans fewer years.")
            
            # trend_chart = create_popularity_trend_chart(search_df, search_name, sex, type='rank')
            # if trend_chart:
            #     st.plotly_chart(trend_chart, use_container_width=True)
            
        else:
            st.warning(f"No data found for '{search_name}' ({sex})")

def name_origin_page():
    """Name origin page"""
    st.title("🌍 Name Origin")
    st.markdown("Explore the origin of a name")
    # TODO: Use the API for https://namsor.app/ to get the origin of a name
    

def get_groq_client():
    """Get Groq client with API key from environment or secrets"""
    api_key = None
    try:
        # Try Streamlit secrets first
        api_key = st.secrets.get("GROQ_API_KEY")
    except:
        pass
    
    if not api_key:
        # Fall back to environment variable
        api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("GROQ_API_KEY not found. Please set it in environment variables or .streamlit/secrets.toml")
        return None
    
    return Groq(api_key=api_key)


def validate_sql_query(query):
    """Validate SQL query for safety - only SELECT statements allowed"""
    if not query:
        return False, "Empty query"
    
    # Strip whitespace and convert to uppercase for checking
    query_upper = query.strip().upper()
    
    # Check that it starts with SELECT
    if not query_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed"
    
    # Check for dangerous keywords
    dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False, f"Query contains forbidden keyword: {keyword}"
    
    # Check for LIMIT clause
    if "LIMIT" not in query_upper:
        # Add LIMIT if not present
        # Try to find where to add it (before any semicolon or at the end)
        if query.rstrip().endswith(";"):
            query = query.rstrip()[:-1].rstrip() + " LIMIT 1000;"
        else:
            query = query.rstrip() + " LIMIT 1000"
    
    return True, query


def execute_safe_sql(query, max_rows=1000):
    """Execute a SQL query safely with row limit"""
    # Validate query first
    is_valid, result = validate_sql_query(query)
    if not is_valid:
        return None, result
    
    # Use the validated query (may have LIMIT added)
    query = result
    
    # Ensure LIMIT doesn't exceed max_rows
    query_upper = query.upper()
    limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
    if limit_match:
        limit_value = int(limit_match.group(1))
        if limit_value > max_rows:
            query = re.sub(r'LIMIT\s+\d+', f'LIMIT {max_rows}', query, flags=re.IGNORECASE)
    else:
        # Add LIMIT if somehow missing
        if query.rstrip().endswith(";"):
            query = query.rstrip()[:-1].rstrip() + f" LIMIT {max_rows};"
        else:
            query = query.rstrip() + f" LIMIT {max_rows}"
    
    try:
        conn = sqlite3.connect("data/names.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Double-check row limit
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        return df, None
    except Exception as e:
        return None, str(e)


def generate_sql_from_question(question, conversation_history):
    """Use Groq to generate SQL query from natural language question"""
    client = get_groq_client()
    if not client:
        return None, "Groq client not available"
    
    # Database schema information
    schema_context = """
Database Schema:
- Table: names
- Columns:
  - name (TEXT): Baby name
  - sex (TEXT): 'M' for Male or 'F' for Female
  - total_count (INTEGER): Number of babies with this name in the given year
  - year (INTEGER): Year (1880-2024)
  - popularity_percent (REAL): Percentage of babies with this name for the given sex/year
  - popularity_rank (INTEGER): Ranking of the name for the given sex/year (1 = most popular)

Important Guidelines:
- Prefer aggregation queries with GROUP BY, SUM, COUNT, AVG, etc. when summarizing data
- Always include a LIMIT clause (max 1000 rows)
- Use appropriate WHERE clauses to filter data
- For name searches, use LOWER() function for case-insensitive matching
"""
    
    # Build conversation context
    messages = [
        {
            "role": "system",
            "content": f"""You are a SQL query generator. Your task is to translate natural language questions about a baby names database into SQL queries.

{schema_context}

Rules:
1. Only generate SELECT queries
2. Always include LIMIT 1000 in your queries
3. Prefer using aggregations (GROUP BY, SUM, COUNT, AVG, MAX, MIN) to summarize data rather than returning large result sets
4. Return only the SQL query, no explanation or markdown formatting
5. Use proper SQL syntax for SQLite
"""
        }
    ]
    
    # Add conversation history for context
    if conversation_history:
        for entry in conversation_history[-4:]:  # Last 4 exchanges for context
            if "user" in entry:
                messages.append({"role": "user", "content": entry["user"]})
            if "sql" in entry:
                messages.append({"role": "assistant", "content": f"SQL: {entry['sql']}"})
    
    # Add current question
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Clean up the SQL query (remove markdown code blocks if present)
        sql_query = re.sub(r'^```sql\s*', '', sql_query)
        sql_query = re.sub(r'^```\s*', '', sql_query)
        sql_query = re.sub(r'\s*```\s*$', '', sql_query)
        sql_query = sql_query.strip()
        
        return sql_query, None
    except Exception as e:
        return None, str(e)


def generate_answer_from_results(question, sql_query, query_results, conversation_history):
    """Use Groq to generate natural language answer from SQL query results"""
    client = get_groq_client()
    if not client:
        return None, "Groq client not available"
    
    # Format query results as text
    if query_results is None or query_results.empty:
        results_text = "No results returned from the query."
    else:
        # Convert DataFrame to a readable format
        results_text = query_results.to_string(index=False)
        if len(results_text) > 5000:  # Truncate if too long
            results_text = results_text[:5000] + "\n... (truncated)"
    
    # Build conversation context
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that answers questions about baby names data from the Social Security Administration database. 
You analyze SQL query results and provide clear, concise answers to user questions.
Always format numbers nicely (e.g., use commas for large numbers).
Be conversational and helpful, but stay accurate to the data.
"""
        }
    ]
    
    # Add conversation history for context
    if conversation_history:
        for entry in conversation_history[-4:]:  # Last 4 exchanges for context
            if "user" in entry:
                messages.append({"role": "user", "content": entry["user"]})
            if "assistant" in entry:
                messages.append({"role": "assistant", "content": entry["assistant"]})
    
    # Add current question and results
    messages.append({
        "role": "user",
        "content": f"""Question: {question}

SQL Query Executed:
{sql_query}

Query Results:
{results_text}

Please answer the user's question based on the query results above. Be concise and helpful."""
    })
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        return answer, None
    except Exception as e:
        return None, str(e)


def chatbot_page():
    """AI Chatbot page for querying the names database"""
    st.title("💬 AI Chatbot")
    st.markdown("Ask questions about the baby names database in natural language. The AI will translate your questions into SQL queries and provide answers based on the data.")
    
    # Suggested prompts
    suggested_prompts = [
        "What years do you have baby name data for?",
        "What were the top 10 most popular boy names in 2024?",
        "What are the most popular baby names over the past 10 years?",
        "What popularity is the name Oliver currently at?",
        "Is the name David increasing in popularity?",
        "Which baby names have risen in popularity the fastest over the past 10 years?"
    ]
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Show suggested prompts if no chat history
    if len(st.session_state.chat_history) == 0:
        st.markdown("### 💡 Try asking:")
        # Display prompts in a grid layout
        cols = st.columns(2)
        for idx, prompt in enumerate(suggested_prompts):
            col = cols[idx % 2]
            if col.button(prompt, key=f"suggest_{idx}", use_container_width=True):
                # Add the prompt as if user typed it
                st.session_state.chat_history.append({"user": prompt})
                st.rerun()
    
    # Display chat history
    for entry in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.write(entry["user"])
        
        # Assistant response with SQL query
        with st.chat_message("assistant"):
            # Display SQL query in expandable section if available
            if "sql" in entry and entry["sql"]:
                with st.expander("📊 View SQL Query", expanded=False):
                    st.code(entry["sql"], language="sql")
            
            # Display assistant answer if available
            if "assistant" in entry:
                st.write(entry["assistant"])
            
            # Display error if any
            if "error" in entry and entry["error"]:
                st.error(f"Error: {entry['error']}")
                # Show raw results if available as fallback
                if "raw_results" in entry and entry["raw_results"] is not None:
                    st.write("Query Results:")
                    st.dataframe(entry["raw_results"], use_container_width=True)
    
    # Check if there's an unprocessed message (from button click)
    if (len(st.session_state.chat_history) > 0 and 
        "assistant" not in st.session_state.chat_history[-1] and 
        "user" in st.session_state.chat_history[-1]):
        # Process the query from button click
        prompt = st.session_state.chat_history[-1]["user"]
        
        # Process the query
        # Generate SQL query
        sql_query, sql_error = generate_sql_from_question(prompt, st.session_state.chat_history)
        
        if sql_error:
            st.session_state.chat_history[-1]["sql"] = None
            st.session_state.chat_history[-1]["assistant"] = "I couldn't generate a SQL query. Please try rephrasing your question."
            st.session_state.chat_history[-1]["error"] = sql_error
            st.rerun()
        
        # Execute SQL query
        query_results, exec_error = execute_safe_sql(sql_query)
        
        if exec_error:
            st.session_state.chat_history[-1]["sql"] = sql_query
            st.session_state.chat_history[-1]["assistant"] = "I encountered an error executing the SQL query."
            st.session_state.chat_history[-1]["error"] = exec_error
            st.rerun()
        
        # Generate answer from results
        answer, answer_error = generate_answer_from_results(
            prompt, sql_query, query_results, st.session_state.chat_history
        )
        
        if answer_error:
            st.session_state.chat_history[-1]["sql"] = sql_query
            st.session_state.chat_history[-1]["assistant"] = "I encountered an error generating the answer."
            st.session_state.chat_history[-1]["error"] = answer_error
            # Show raw results as fallback in the error entry
            if query_results is not None and not query_results.empty:
                st.session_state.chat_history[-1]["raw_results"] = query_results
            st.rerun()
        
        # Update history with successful response
        st.session_state.chat_history[-1]["sql"] = sql_query
        st.session_state.chat_history[-1]["assistant"] = answer
        st.session_state.chat_history[-1]["error"] = None
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about baby names..."):
        # Add user message to history
        st.session_state.chat_history.append({"user": prompt})
        
        # Process the query
        # Generate SQL query
        sql_query, sql_error = generate_sql_from_question(prompt, st.session_state.chat_history)
        
        if sql_error:
            st.session_state.chat_history[-1]["sql"] = None
            st.session_state.chat_history[-1]["assistant"] = "I couldn't generate a SQL query. Please try rephrasing your question."
            st.session_state.chat_history[-1]["error"] = sql_error
            st.rerun()
        
        # Execute SQL query
        query_results, exec_error = execute_safe_sql(sql_query)
        
        if exec_error:
            st.session_state.chat_history[-1]["sql"] = sql_query
            st.session_state.chat_history[-1]["assistant"] = "I encountered an error executing the SQL query."
            st.session_state.chat_history[-1]["error"] = exec_error
            st.rerun()
        
        # Generate answer from results
        answer, answer_error = generate_answer_from_results(
            prompt, sql_query, query_results, st.session_state.chat_history
        )
        
        if answer_error:
            st.session_state.chat_history[-1]["sql"] = sql_query
            st.session_state.chat_history[-1]["assistant"] = "I encountered an error generating the answer."
            st.session_state.chat_history[-1]["error"] = answer_error
            # Show raw results as fallback in the error entry
            if query_results is not None and not query_results.empty:
                st.session_state.chat_history[-1]["raw_results"] = query_results
            st.rerun()
        
        # Update history with successful response
        st.session_state.chat_history[-1]["sql"] = sql_query
        st.session_state.chat_history[-1]["assistant"] = answer
        st.session_state.chat_history[-1]["error"] = None
        st.rerun()


def main():
    """Main function with page navigation"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "💬 AI Chatbot", "🔍 Name Search", "👶 Top Names"])
    
    # Route to appropriate page
    if page == "🏠 Home":
        main_page()
    elif page == "👶 Top Names":
        top_names_page()
    elif page == "🔍 Name Search":
        search_page()
    elif page == "💬 AI Chatbot":
        chatbot_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Data source: [Social Security Administration](https://www.ssa.gov/oact/babynames/limits.html)")
    st.sidebar.markdown("Built with Streamlit and Plotly")

if __name__ == "__main__":
    main()