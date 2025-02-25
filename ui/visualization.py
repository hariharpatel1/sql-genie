import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

def determine_chart_type(df: pd.DataFrame, query_understanding: Dict[str, Any]) -> str:
    """
    Determine the most appropriate chart type based on the data and query.
    
    Args:
        df: DataFrame with query results
        query_understanding: Dict with information about the query intent
        
    Returns:
        String indicating the chart type
    """
    # Get column data types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' 
                or pd.api.types.is_datetime64_any_dtype(df[col])
                or 'date' in col.lower() 
                or 'time' in col.lower() 
                or 'month' in col.lower()
                or 'year' in col.lower()]
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Extract query intent if available
    intent = query_understanding.get('intent', '').lower() if query_understanding else ''
    
    # Check for time series data
    if len(date_cols) >= 1 and len(numeric_cols) >= 1:
        return 'line'
    
    # Check for comparisons between categories
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        if len(df) > 10:
            return 'bar'
        else:
            return 'bar'
    
    # Check for distributions
    if len(numeric_cols) >= 1 and 'distribution' in intent:
        return 'histogram'
    
    # Check for correlations
    if len(numeric_cols) >= 2 and 'correlation' in intent:
        return 'scatter'
    
    # Check for proportions or parts of a whole
    if 'percentage' in intent or 'proportion' in intent or 'share' in intent:
        return 'pie' if len(df) <= 7 else 'bar'
    
    # Default to bar for most queries
    if len(numeric_cols) >= 1:
        return 'bar'
    
    # If no numeric columns, return None
    return 'table'

def setup_chart_aesthetics(fig, ax, title=None, customized=False):
    """
    Apply aesthetic improvements to the chart
    
    Args:
        fig: matplotlib figure
        ax: matplotlib axis
        title: Chart title
        customized: Whether to apply more custom styling
    """
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # Set spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if customized:
        # Use a custom color palette
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("viridis", 10))
        
        # Improve grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust background
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')
        
        # Enhance fonts
        plt.rcParams['font.family'] = 'sans-serif'
        
        # Add subtle shadow to the figure
        fig.patch.set_alpha(0.9)
    else:
        # Lighter styling
        ax.grid(axis='y', linestyle=':', alpha=0.3)

def create_visualization(df: pd.DataFrame, query_understanding: Dict[str, Any] = None) -> Optional[plt.Figure]:
    """
    Create a basic visualization based on query results
    
    Args:
        df: DataFrame with query results
        query_understanding: Dict with information about the query intent
        
    Returns:
        matplotlib Figure or None if visualization couldn't be created
    """
    if df.empty or len(df.columns) < 2:
        return None
    
    try:
        # Determine the appropriate chart type
        chart_type = determine_chart_type(df, query_understanding)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get column data types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
                    or 'date' in col.lower() 
                    or 'time' in col.lower() 
                    or 'month' in col.lower()
                    or 'year' in col.lower()]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Set chart title
        title = query_understanding.get('title', 'Query Results Visualization') if query_understanding else 'Query Results Visualization'
        
        # Create the appropriate visualization
        if chart_type == 'line':
            # For line charts, prefer date on x-axis and numeric on y-axis
            x_col = date_cols[0] if date_cols else (categorical_cols[0] if categorical_cols else df.columns[0])
            y_col = numeric_cols[0] if numeric_cols else df.columns[1]
            
            df.plot(x=x_col, y=y_col, kind='line', ax=ax, marker='o', markersize=5)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            
        elif chart_type == 'bar':
            # For bar charts, prefer categorical on x-axis and numeric on y-axis
            x_col = categorical_cols[0] if categorical_cols else (date_cols[0] if date_cols else df.columns[0])
            y_col = numeric_cols[0] if numeric_cols else df.columns[1]
            
            # Limit to top 15 entries if there are too many
            if len(df) > 15:
                top_df = df.nlargest(15, y_col)
                top_df.plot(x=x_col, y=y_col, kind='bar', ax=ax)
                title += " (Top 15)"
            else:
                df.plot(x=x_col, y=y_col, kind='bar', ax=ax)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'pie':
            # For pie charts, use the first categorical and numeric columns
            label_col = categorical_cols[0] if categorical_cols else df.columns[0]
            value_col = numeric_cols[0] if numeric_cols else df.columns[1]
            
            # Create a pie chart with percentages
            df.plot(y=value_col, kind='pie', labels=df[label_col], autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            
        elif chart_type == 'histogram':
            # For histograms, use the first numeric column
            value_col = numeric_cols[0] if numeric_cols else df.columns[0]
            
            df[value_col].plot(kind='hist', bins=10, ax=ax)
            ax.set_xlabel(value_col)
            ax.set_ylabel('Frequency')
            
        elif chart_type == 'scatter':
            # For scatter plots, use the first two numeric columns
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                df.plot(x=x_col, y=y_col, kind='scatter', ax=ax)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
        
        # Apply aesthetics
        setup_chart_aesthetics(fig, ax, title)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None

def create_advanced_visualization(df: pd.DataFrame, query_understanding: Dict[str, Any] = None, alt_style: bool = False) -> Optional[plt.Figure]:
    """
    Create an advanced visualization with more styling and features
    
    Args:
        df: DataFrame with query results
        query_understanding: Dict with information about the query intent
        alt_style: Use alternative styling
        
    Returns:
        matplotlib Figure or None if visualization couldn't be created
    """
    if df.empty or len(df.columns) < 2:
        return None
    
    try:
        # Set seaborn style
        sns.set_style("whitegrid" if not alt_style else "darkgrid")
        
        # Determine the appropriate chart type
        chart_type = determine_chart_type(df, query_understanding)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get column data types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
                    or 'date' in col.lower() 
                    or 'time' in col.lower() 
                    or 'month' in col.lower()
                    or 'year' in col.lower()]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Set chart title
        title = query_understanding.get('title', 'Advanced Data Visualization') if query_understanding else 'Advanced Data Visualization'
        
        # Create a custom colormap for variety
        if alt_style:
            colors = sns.color_palette("magma", 10)
        else:
            colors = sns.color_palette("viridis", 10)
            
        # Create the appropriate visualization with advanced styling
        if chart_type == 'line':
            # For line charts, prefer date on x-axis and numeric on y-axis
            x_col = date_cols[0] if date_cols else (categorical_cols[0] if categorical_cols else df.columns[0])
            
            # If multiple numeric columns, plot them all
            if len(numeric_cols) > 1 and len(numeric_cols) <= 5:
                for i, y_col in enumerate(numeric_cols[:5]):
                    sns.lineplot(x=x_col, y=y_col, data=df, marker='o', label=y_col, color=colors[i % len(colors)], ax=ax)
            else:
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                sns.lineplot(x=x_col, y=y_col, data=df, marker='o', color=colors[0], ax=ax)
                
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col if 'y_col' in locals() else 'Value')
            
        elif chart_type == 'bar':
            # For bar charts, prefer categorical on x-axis and numeric on y-axis
            x_col = categorical_cols[0] if categorical_cols else (date_cols[0] if date_cols else df.columns[0])
            y_col = numeric_cols[0] if numeric_cols else df.columns[1]
            
            # Limit to top entries if there are too many
            if len(df) > 12:
                top_df = df.nlargest(12, y_col)
                # Use seaborn for better styling
                sns.barplot(x=x_col, y=y_col, data=top_df, palette=colors, ax=ax)
                title += " (Top 12)"
            else:
                sns.barplot(x=x_col, y=y_col, data=df, palette=colors, ax=ax)
            
            # Add value labels on top of bars
            for i, p in enumerate(ax.patches):
                value = p.get_height()
                ax.annotate(f'{value:.1f}', 
                            (p.get_x() + p.get_width() / 2., value),
                            ha='center', va='bottom', fontsize=9, rotation=0,
                            xytext=(0, 5), textcoords='offset points')
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'pie':
            # For pie charts, use the first categorical and numeric columns
            label_col = categorical_cols[0] if categorical_cols else df.columns[0]
            value_col = numeric_cols[0] if numeric_cols else df.columns[1]
            
            # Create a pie chart with percentages and a donut shape
            wedges, texts, autotexts = ax.pie(
                df[value_col], 
                labels=df[label_col], 
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'width': 0.5, 'edgecolor': 'w', 'linewidth': 2},
                textprops={'fontsize': 10},
                colors=colors
            )
            
            # Improve text visibility
            plt.setp(autotexts, size=9, weight="bold", color="white")
            ax.set_ylabel('')
            
            # Add center circle for donut effect
            centre_circle = plt.Circle((0, 0), 0.3, fc='white')
            fig.gca().add_artist(centre_circle)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_aspect('equal')
            
        elif chart_type == 'histogram':
            # For histograms, use the first numeric column
            value_col = numeric_cols[0] if numeric_cols else df.columns[0]
            
            # Use KDE for a density curve
            sns.histplot(df[value_col], kde=True, color=colors[0], alpha=0.7, ax=ax)
            ax.set_xlabel(value_col)
            ax.set_ylabel('Frequency')
            
            # Add vertical line for mean
            mean_val = df[value_col].mean()
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, alpha=0.8)
            ax.text(mean_val, ax.get_ylim()[1]*0.9, f'Mean: {mean_val:.2f}', 
                    horizontalalignment='center', color='darkred')
            
        elif chart_type == 'scatter':
            # For scatter plots, use the first two numeric columns
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                # If we have a third column, use it for size
                if len(numeric_cols) >= 3:
                    z_col = numeric_cols[2]
                    size_vals = df[z_col] * 100 / df[z_col].max()
                    
                    # Use seaborn's scatterplot with size variations
                    sns.scatterplot(x=x_col, y=y_col, size=z_col, hue=z_col,
                                  data=df, sizes=(20, 200), palette='viridis', ax=ax)
                    
                    # Add a size legend
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, title=z_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # Use simple scatterplot with a colorful palette
                    if len(categorical_cols) > 0:
                        hue_col = categorical_cols[0]
                        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, palette='viridis', s=100, ax=ax)
                    else:
                        sns.scatterplot(x=x_col, y=y_col, data=df, color=colors[0], s=100, ax=ax)
                    
                    # Add trendline
                    sns.regplot(x=x_col, y=y_col, data=df, scatter=False, ax=ax, color='red', line_kws={"linestyle": "--"})
                
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
        
        elif chart_type == 'table':
            # If no good visualization is possible, show a styled table
            ax.axis('off')
            
            # Create a table with data from the first few rows and columns
            table_data = df.head(10).values
            col_labels = df.columns
            
            table = ax.table(cellText=table_data, colLabels=col_labels, 
                           loc='center', cellLoc='center')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color the header
            for i, key in enumerate(col_labels):
                cell = table[(0, i)]
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(colors[0])
        
        # Apply aesthetics
        setup_chart_aesthetics(fig, ax, title, customized=True)
        
        # Add a subtle shadow to the entire figure
        if alt_style:
            fig.patch.set_alpha(0.8)
            fig.patch.set_facecolor('#F0F0F0')
        
        # Add a watermark
        ax.text(0.99, 0.01, 'SQL Genie 🧞', fontsize=8, color='gray',
               ha='right', va='bottom', transform=ax.transAxes, alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating advanced visualization: {str(e)}")
        return None