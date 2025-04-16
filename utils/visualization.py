import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def generate_visualizations(df, title=None, colorscale="Viridis", height=600, width=800, template="plotly_white"):
    """
    Generate a heatmap visualization for numeric data.

    Args:
        df (pd.DataFrame): Input DataFrame.
        title (str, optional): Title for the plot.
        colorscale (str, optional): Plotly colorscale for the heatmap.

    Returns:
        go.Figure: Plotly Figure object.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame) or df is None:
        raise ValueError("Input must be a non-empty pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame must contain at least one numeric column.")

    # Use only numeric columns for heatmap
    heatmap_data = df[numeric_cols]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=colorscale
        )
    )
    fig.update_layout(
        title=title or "Data Heatmap",
        xaxis_title="Columns",
        yaxis_title="Index",
        height=height,
        width=width
    )
    return fig

def create_advanced_visualization(df, x_col, y_col, color_col=None, title=None, colorscale="Viridis", template="plotly_white"):
    """
    Create an advanced scatter plot with optional color grouping.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        color_col (str, optional): Column name for grouping/color.
        title (str, optional): Plot title.
        colorscale (str, optional): Plotly colorscale.

    Returns:
        go.Figure: Plotly Figure object.
    """
    if not isinstance(df, pd.DataFrame) or df is None:
        raise ValueError("Input must be a non-empty pandas DataFrame.")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("x_col and y_col must be valid column names in the DataFrame.")

    if color_col and color_col in df.columns:
        # Map categories to integers for color
        categories = df[color_col].astype('category')
        color_vals = categories.cat.codes
        fig = go.Figure(
            data=go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(
                    color=color_vals,
                    colorscale=colorscale,
                    colorbar=dict(
                        title=color_col,
                        tickvals=list(range(len(categories.cat.categories))),
                        ticktext=list(categories.cat.categories)
                    )
                ),
                text=df[color_col].astype(str),
                showlegend=False
            )
        )
        fig.update_layout(title=title or f"{y_col} vs {x_col} by {color_col}")
    else:
        fig = go.Figure(
            data=go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(colorscale=colorscale)
            )
        )
        fig.update_layout(title=title or f"{y_col} vs {x_col}")

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template=template
    )
    return fig


def create_group_visualization(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_cols: List[str],
    numeric_cols: list,
    viz_type: str = "box"
) -> go.Figure:
    """
    Create visualization for grouped data analysis
    
    Args:
        df: Input DataFrame
        group_cols: Columns to group by
        agg_cols: Columns to aggregate
        viz_type: Type of visualization ('bar' or 'box')
    
    Returns:
        Plotly figure object
    """
    try:
        if not group_cols or not agg_cols:
            raise ValueError("Both group_cols and agg_cols must be provided")
        
        if viz_type == "bar":
            # Create bar chart
            grouped_data = df.groupby(group_cols)[agg_cols].mean().reset_index()
            fig = px.bar(
                grouped_data,
                x=group_cols[0],
                y=agg_cols[0],
                color=group_cols[1] if len(group_cols) > 1 else None,
                barmode='group',
                title=f"Average {agg_cols[0]} by {', '.join(group_cols)}",
                template=template,
                color_continuous_scale=colorscale.lower()
            )
        elif viz_type == "box":
            # Create box plot
            fig = px.box(
                df,
                x=group_cols[0],
                y=agg_cols[0],
                color=group_cols[1] if len(group_cols) > 1 else None,
                title=f"Distribution of {agg_cols[0]} by {', '.join(group_cols)}",
                template=template,
                color_continuous_scale=colorscale.lower()
            )
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        # Update layout
        fig.update_layout(
            xaxis_title=group_cols[0],
            yaxis_title=agg_cols[0],
            template=template,
            showlegend=True
        )
            
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create group visualization: {str(e)}")
        raise RuntimeError(f"Visualization failed: {str(e)}")