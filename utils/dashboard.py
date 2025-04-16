import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class InteractiveDashboard:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        self.date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    def create_time_series_plot(self, x_col: str, y_col: str, groupby: str = None, colorscale: str = "Cividis", template: str = "plotly_white") -> go.Figure:
        """Create interactive time series plot"""
        try:
            color_map = {
                "cividis": "#3b518a",
                "viridis": "#440154",
                "plasma": "#0d0887",
                "inferno": "#000004",
                "magma": "#000004",
                "jet": "#00007F",
                "hot": "#FF0000",
                "cool": "#00FFFF",
            }
            selected_color = color_map.get(colorscale.lower(), "#3b518a")
            if groupby:
                fig = px.line(
                    self.df, x=x_col, y=y_col, color=groupby,
                    title=f'{y_col} over {x_col} by {groupby}',
                    color_discrete_sequence=[selected_color],
                    template=template
                )
            else:
                fig = px.line(
                    self.df, x=x_col, y=y_col,
                    title=f'{y_col} over {x_col}',
                    color_discrete_sequence=[selected_color],
                    template=template
                )
            fig.update_layout(
                hovermode='x unified',
                xaxis_title=x_col,
                yaxis_title=y_col
            )
            return fig
        except Exception as e:
            logger.error(f"Failed to create time series plot: {str(e)}")
            raise

    def create_distribution_plot(self, column: str, colorscale: str = "Cividis", template: str = "plotly_white") -> go.Figure:
        """Create interactive distribution plot"""
        try:
            # Map colorscale name to a representative color for marker_color
            color_map = {
                "cividis": "#3b518a",
                "viridis": "#440154",
                "plasma": "#0d0887",
                "inferno": "#000004",
                "magma": "#000004",
                "jet": "#00007F",
                "hot": "#FF0000",
                "cool": "#00FFFF",
            }
            selected_color = color_map.get(colorscale.lower(), "#3b518a")
    
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Distribution', 'Box Plot'))
    
            # Add histogram
            fig.add_trace(
                go.Histogram(x=self.df[column], name="Distribution", marker_color=selected_color),
                row=1, col=1
            )
    
            # Add box plot
            fig.add_trace(
                go.Box(x=self.df[column], name="Box Plot", marker_color=selected_color),
                row=2, col=1
            )
    
            fig.update_layout(
                title=f'Distribution Analysis: {column}',
                showlegend=False,
                height=600,
                template=template
            )
            return fig
        except Exception as e:
            logger.error(f"Failed to create distribution plot: {str(e)}")
            raise

    def create_correlation_matrix(self, colorscale: str = "Cividis", template: str = "plotly_white") -> go.Figure:
        """Create interactive correlation matrix"""
        try:
            corr_matrix = self.df[self.numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title='Correlation Matrix',
                aspect='auto',
                color_continuous_scale=colorscale.lower(),
                template=template
            )
            fig.update_traces(text=corr_matrix.round(2), texttemplate='%{text}')
            return fig
        except Exception as e:
            logger.error(f"Failed to create correlation matrix: {str(e)}")
            raise

    def create_scatter_matrix(self, columns: List[str], color: str = None, colorscale: str = "Cividis", template: str = "plotly_white") -> go.Figure:
        try:
            fig = px.scatter_matrix(
                self.df[columns],
                color=self.df[color] if color else None,
                title='Scatter Matrix',
                opacity=0.5,
                color_continuous_scale=colorscale.lower() if color else None,
                template=template
            )
            return fig
        except Exception as e:
            logger.error(f"Failed to create scatter matrix: {str(e)}")
            raise