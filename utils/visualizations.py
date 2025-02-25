import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def create_visualization(df: pd.DataFrame, query_understanding: Dict[str, Any] = None) -> Optional[plt.Figure]:
    """
    Create a visualization based on the query results and understanding
    
    Args:
        df: DataFrame with query results
        query_understanding: Dictionary with query understanding (optional)
        
    Returns:
        Matplotlib figure or None if visualization not possible
    """
    if df is None or df.empty or len(df) < 2:  # Need at least 2 rows for most visualizations
        return None
    
    try:
        # Set default style
        sns.set_style("whitegrid")
        
        # Determine what visualization would be appropriate
        viz_type = determine_visualization_type(df, query_understanding)
        
        if viz_type == "time_series":
            return create_time_series(df, query_understanding)
        elif viz_type == "bar":
            return create_bar_chart(df, query_understanding)
        elif viz_type == "pie":
            return create_pie_chart(df, query_understanding)
        elif viz_type == "histogram":
            return create_histogram(df, query_understanding)
        elif viz_type == "scatter":
            return create_scatter_plot(df, query_understanding)
        else:
            # Default to a data table visualization
            return None
    
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None

def determine_visualization_type(df: pd.DataFrame, query_understanding: Dict[str, Any] = None) -> str:
    """
    Determine the most appropriate visualization type for the data
    
    Args:
        df: DataFrame with query results
        query_understanding: Dictionary with query understanding
        
    Returns:
        String with visualization type
    """
    if query_understanding is None:
        query_understanding = {}
    
    # Check for time series data
    has_date_col = any(is_date_column(df, col) for col in df.columns)
    
    # Check for categorical and numerical columns
    cat_cols = [col for col in df.columns if is_categorical_column(df, col)]
    num_cols = [col for col in df.columns if is_numerical_column(df, col)]
    
    # Get aggregations from query understanding
    aggregations = query_understanding.get("aggregations", [])
    
    # Decision logic for visualization type
    if has_date_col and any(num_cols) and len(df) > 3:
        return "time_series"
    elif cat_cols and num_cols and len(df) <= 20 and len(df) > 1:
        if len(cat_cols) == 1 and len(df) <= 10 and "sum" in str(aggregations).lower():
            return "pie"
        else:
            return "bar"
    elif len(num_cols) >= 2:
        return "scatter"
    elif num_cols:
        return "histogram"
    else:
        return "table"

def is_date_column(df: pd.DataFrame, column_name: str) -> bool:
    """Check if a column contains date/time data"""
    if df[column_name].dtype in ['datetime64[ns]', 'datetime64']:
        return True
    
    # Try to convert to datetime
    try:
        pd.to_datetime(df[column_name])
        return True
    except:
        return False

def is_categorical_column(df: pd.DataFrame, column_name: str) -> bool:
    """Check if a column contains categorical data"""
    if df[column_name].dtype == 'object' or df[column_name].dtype.name == 'category':
        return True
    
    # If numeric but few unique values relative to size, treat as categorical
    if df[column_name].dtype in ['int64', 'float64'] and df[column_name].nunique() <= min(20, len(df) * 0.5):
        return True
    
    return False

def is_numerical_column(df: pd.DataFrame, column_name: str) -> bool:
    """Check if a column contains numerical data"""
    return df[column_name].dtype in ['int64', 'float64']

def create_time_series(df: pd.DataFrame, query_understanding: Dict[str, Any] = None) -> plt.Figure:
    """Create a time series plot"""
    # Find time column
    time_col = next((col for col in df.columns if is_date_column(df, col)), None)
    
    # Find numerical column(s)
    num_cols = [col for col in df.columns if is_numerical_column(df, col)]
    
    if not time_col or not num_cols:
        return None
    
    # Ensure datetime type
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each numerical column
    for col in num_cols[:3]:  # Limit to 3 lines for readability
        ax.plot(df[time_col], df[col], marker='o', linestyle='-', label=col)
    
    # Add labels and legend
    ax.set_title(f"Time Series Analysis")
    ax.set_xlabel(time_col)
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
    
    # Format x-axis date labels
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    return fig

def create_bar_chart(df: pd.DataFrame, query_understanding: Dict[str, Any] = None) -> plt.Figure:
    """Create a bar chart"""
    # Find categorical column
    cat_cols = [col for col in df.columns if is_categorical_column(df, col)]
    
    # Find numerical column
    num_cols = [col for col in df.columns if is_numerical_column(df, col)]
    
    if not cat_cols or not num_cols:
        return None
    
    cat_col = cat_cols[0]
    num_col = num_cols[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    if len(df) > 10:
        # Horizontal bar chart for many categories
        df_sorted = df.sort_values(by=num_col, ascending=False).head(10)
        ax.barh(df_sorted[cat_col], df_sorted[num_col])
        ax.set_xlabel(num_col)
        ax.set_ylabel(cat_col)
    else:
        # Vertical bar chart for few categories
        ax.bar(df[cat_col], df[num_col])
        ax.set_xlabel(cat_col)
        ax.set_ylabel(num_col)
        plt.xticks(rotation=45, ha='right')
    
    ax.set_title(f"{num_col} by {cat_col}")
    plt.tight_layout()
    
    return fig

def create_pie_chart(df: pd.DataFrame, query_understanding: Dict[str, Any] = None) -> plt.Figure:
    """Create a pie chart"""
    # Find categorical column
    cat_cols = [col for col in df.columns if is_categorical_column(df, col)]
    
    # Find numerical column
    num_cols = [col for col in df.columns if is_numerical_column(df, col)]
    
    if not cat_cols or not num_cols:
        return None
    
    cat_col = cat_cols[0]
    num_col = num_cols[0]
    
    # Ensure we're not plotting too many slices
    if df[cat_col].nunique() > 10:
        # Too many categories, get top 9 and group others
        top_values = df.nlargest(9, num_col)[cat_col].unique()
        df_grouped = pd.DataFrame()
        df_grouped[num_col] = [
            df[df[cat_col].isin(top_values)][num_col].sum(),
            df[~df[cat_col].isin(top_values)][num_col].sum()
        ]
        df_grouped[cat_col] = ['Top Categories', 'Others']
        df = df_grouped
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        df[num_col], 
        labels=df[cat_col], 
        autopct='%1.1f%%',
        textprops={'fontsize': 9},
        startangle=90
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add title
    ax.set_title(f"Distribution of {num_col} by {cat_col}")
    
    # Add legend if there are many categories
    if len(df) > 5:
        ax.legend(
            wedges, 
            df[cat_col],
            title=cat_col,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
    
    plt.tight_layout()
    
    return fig

def create_histogram(df: pd.DataFrame, query_understanding: Dict[str, Any] = None) -> plt.Figure:
    """Create a histogram for numerical data"""
    # Find numerical columns
    num_cols = [col for col in df.columns if is_numerical_column(df, col)]
    
    if not num_cols:
        return None
    
    # Use the first numerical column
    num_col = num_cols[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate number of bins using Freedman-Diaconis rule
    q75, q25 = np.percentile(df[num_col], [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(df) ** (1/3)) if iqr > 0 else 'auto'
    bins = int((df[num_col].max() - df[num_col].min()) / bin_width) if bin_width != 'auto' else 'auto'
    
    # Create histogram
    sns.histplot(df[num_col], bins=bins, kde=True, ax=ax)
    
    # Add titles and labels
    ax.set_title(f"Distribution of {num_col}")
    ax.set_xlabel(num_col)
    ax.set_ylabel("Frequency")
    
    # Add vertical line for mean and median
    mean_value = df[num_col].mean()
    median_value = df[num_col].median()
    
    ax.axvline(mean_value, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_value:.2f}')
    ax.axvline(median_value, color='g', linestyle='-.', alpha=0.7, label=f'Median: {median_value:.2f}')
    
    ax.legend()
    plt.tight_layout()
    
    return fig

def create_scatter_plot(df: pd.DataFrame, query_understanding: Dict[str, Any] = None) -> plt.Figure:
    """Create a scatter plot for two numerical columns"""
    # Find numerical columns
    num_cols = [col for col in df.columns if is_numerical_column(df, col)]
    
    if len(num_cols) < 2:
        return None
    
    # Use the first two numerical columns
    x_col = num_cols[0]
    y_col = num_cols[1]
    
    # Check if we have a categorical column to use for color
    cat_cols = [col for col in df.columns if is_categorical_column(df, col)]
    color_col = cat_cols[0] if cat_cols and df[cat_cols[0]].nunique() <= 10 else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    if color_col:
        # Use categorical column for coloring
        for category, group in df.groupby(color_col):
            ax.scatter(group[x_col], group[y_col], label=category, alpha=0.7)
        ax.legend(title=color_col)
    else:
        # Simple scatter plot
        ax.scatter(df[x_col], df[y_col], alpha=0.7)
    
    # Add titles and labels
    ax.set_title(f"Relationship between {x_col} and {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    # Add trend line
    try:
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.7)
    except:
        pass
    
    plt.tight_layout()
    
    return fig