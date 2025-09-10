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
    """Create a trend chart showing popularity over years"""
    # Sort by year
    raw_df = raw_df.sort_values('year')
    
    # Create line chart
    fig = go.Figure()
    
    if type == 'percent':
        y = raw_df['popularity_percent']
    else:
        y = raw_df['popularity_rank']

    fig.add_trace(go.Scatter(
        x=raw_df['year'],
        y=y,
        mode='lines+markers',
        name=f'{name} Popularity',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"Popularity {type.capitalize()} Trend for {name} ({sex})",
        xaxis_title="Year",
        yaxis_title=f"Popularity {type.capitalize()}",
        height=400,
        hovermode='x unified'
    )

    # TODO: Add a projection for popularity for the next 10 years
    
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

def main_page():
    """A description of the app, it's purpose and how to use it"""
    st.title("👶 Baby Names Popularity Explorer")
    st.markdown("Explore the popularity of baby names from the Social Security Administration dataset")
    st.markdown("This app allows you to search for specific baby names and view their popularity metrics and projections")
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
    st.dataframe(top_names, use_container_width=True)

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
                        use_container_width=True, height=600, hide_index=True,
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
            
            # Bottom row: Popularity trend
            trend_chart = create_popularity_trend_chart(search_df, search_name, sex)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            
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
    

def main():
    """Main function with page navigation"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "👶 Top Names","🔍 Name Search"])
    
    # Route to appropriate page
    if page == "🏠 Home":
        main_page()
    elif page == "👶 Top Names":
        top_names_page()
    elif page == "🔍 Name Search":
        search_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Data source: [Social Security Administration](https://www.ssa.gov/oact/babynames/limits.html)")
    st.sidebar.markdown("Built with Streamlit and Plotly")

if __name__ == "__main__":
    main()