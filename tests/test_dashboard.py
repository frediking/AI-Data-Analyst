import pytest
import pandas as pd
import plotly.graph_objects as go
from utils.dashboard import InteractiveDashboard

@pytest.fixture
def sample_dashboard():
    """Fixture to create a sample dashboard instance"""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'sales': [100, 150, 120, 180, 130],
        'category': ['A', 'B', 'A', 'B', 'A'],
        'metric': [10.5, 15.2, 12.1, 18.4, 13.7]
    })
    return InteractiveDashboard(df)

def test_dashboard_initialization(sample_dashboard):
    """Test dashboard initialization and column detection"""
    assert len(sample_dashboard.numeric_cols) == 2  # sales, metric
    assert len(sample_dashboard.categorical_cols) == 1  # category
    assert len(sample_dashboard.date_cols) == 1  # date

def test_create_time_series_plot(sample_dashboard):
    """Test time series plot generation"""
    fig = sample_dashboard.create_time_series_plot(
        x_col='date',
        y_col='sales',
        group_col='category'
    )
    assert isinstance(fig, go.Figure)
    assert fig.data  # Check if plot has data

def test_create_distribution_plot(sample_dashboard):
    """Test distribution plot generation"""
    fig = sample_dashboard.create_distribution_plot('metric')
    assert isinstance(fig, go.Figure)
    assert fig.data  # Check if plot has data

def test_create_correlation_matrix(sample_dashboard):
    """Test correlation matrix generation"""
    fig = sample_dashboard.create_correlation_matrix()
    assert isinstance(fig, go.Figure)
    assert fig.data  # Check if plot has data

def test_invalid_column_selection(sample_dashboard):
    """Test error handling for invalid column selection"""
    with pytest.raises(ValueError):
        sample_dashboard.create_time_series_plot(
            x_col='nonexistent',
            y_col='sales'
        )

@pytest.mark.parametrize("x_col,y_col,group_col", [
    ('date', 'sales', None),
    ('date', 'metric', 'category'),
    ('date', 'sales', 'category')
])
def test_plot_combinations(sample_dashboard, x_col, y_col, group_col):
    """Test various combinations of plot configurations"""
    fig = sample_dashboard.create_time_series_plot(x_col, y_col, group_col)
    assert isinstance(fig, go.Figure)
    assert fig.data