import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def generate_visualizations(df: pd.DataFrame) -> Optional[go.Figure]:
    """Generate and cache visualizations"""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for visualization")
            return None
            
        # Create correlation matrix using plotly
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title='Correlation Heatmap',
            aspect='auto',
            labels=dict(color="Correlation")
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=600,
            width=800,
            showlegend=True,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        # Add text annotations
        fig.update_traces(
            text=corr_matrix.round(2),
            texttemplate="%{text}"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {str(e)}")
        raise RuntimeError(f"Visualization generation failed: {str(e)}")

def create_advanced_visualization(df: pd.DataFrame) -> None:
    """Create advanced interactive visualizations"""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for visualization")
            return
            
        # Let user select columns
        x_col = st.selectbox("Select X axis:", numeric_cols)
        y_col = st.selectbox("Select Y axis:", 
                           [col for col in numeric_cols if col != x_col])
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=f"{y_col} vs {x_col}",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Advanced visualization failed: {str(e)}")
        st.error("Failed to create visualization. Please check your data.")

