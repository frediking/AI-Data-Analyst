import pytest
import pandas as pd
import plotly.graph_objects as go
from utils.visualization import generate_visualizations, create_advanced_visualization

@pytest.fixture
def sample_numeric_df():
    """Fixture for numeric data testing"""
    return pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10],
        'z': [1, 3, 5, 7, 9]
    })

@pytest.fixture
def sample_mixed_df():
    """Fixture for mixed data types testing"""
    return pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'A', 'B', 'A'],
        'date': pd.date_range('2024-01-01', periods=5),
        'values': [10.5, 15.2, 12.1, 18.4, 13.7]
    })

def test_visualization_generation_numeric(sample_numeric_df):
    """Test visualization generation with numeric data"""
    fig = generate_visualizations(sample_numeric_df)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.layout.title is not None
    assert all(trace.type == 'heatmap' for trace in fig.data)

def test_visualization_generation_mixed(sample_mixed_df):
    """Test visualization generation with mixed data types"""
    fig = generate_visualizations(sample_mixed_df)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Only numeric columns should be included in correlation
    expected_cols = ['numeric', 'values']
    assert all(col in str(fig.data[0]) for col in expected_cols)

def test_advanced_visualization(sample_mixed_df):
    """Test advanced visualization features"""
    fig = create_advanced_visualization(
        df=sample_mixed_df,
        x_col='numeric',
        y_col='values',
        color_col='categorical'
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == 'scatter'
    assert fig.data[0].mode == 'markers'
    assert 'categorical' in str(fig.data[0].marker.color)

@pytest.mark.parametrize("invalid_df", [
    pd.DataFrame(),  # Empty DataFrame
    pd.DataFrame({'cat': ['A', 'B', 'C']}),  # No numeric columns
    None  # None input
])
def test_visualization_generation_invalid_input(invalid_df):
    """Test error handling for invalid inputs"""
    with pytest.raises((ValueError, TypeError)):
        generate_visualizations(invalid_df)

def test_visualization_layout_customization(sample_numeric_df):
    """Test visualization layout customization"""
    fig = generate_visualizations(
        df=sample_numeric_df,
        title="Custom Title",
        height=800,
        width=1000
    )
    
    assert fig.layout.title.text == "Custom Title"
    assert fig.layout.height == 800
    assert fig.layout.width == 1000

def test_visualization_color_scheme(sample_numeric_df):
    """Test color scheme customization"""
    fig = generate_visualizations(
        df=sample_numeric_df,
        colorscale='Viridis'
    )
    
    assert fig.data[0].colorscale == 'Viridis'

def test_large_dataset_visualization():
    """Test visualization performance with larger datasets"""
    large_df = pd.DataFrame({
        'x': range(1000),
        'y': range(1000),
        'z': range(1000)
    })
    
    fig = generate_visualizations(large_df)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == 'heatmap'