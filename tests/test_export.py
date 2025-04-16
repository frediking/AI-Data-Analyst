import pytest
import pandas as pd
import io
import joblib
import json
from utils.export import export_dataset, export_ml_artifacts, export_quality_report

@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample DataFrame for testing"""
    return pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'text': ['a', 'b', 'c', 'd', 'e'],
        'date': pd.date_range('2024-01-01', periods=5)
    })

@pytest.fixture
def sample_ml_model():
    """Fixture to create a dummy ML model for testing"""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    X = [[1], [2], [3], [4], [5]]
    y = [2, 4, 6, 8, 10]
    model.fit(X, y)
    return model

@pytest.fixture
def sample_quality_metrics():
    """Fixture for sample quality metrics matching new structure"""
    return {
        'overall_completeness': 98.5,
        'null_summary': {
            'col1': {'null_count': 0, 'null_percentage': 0.0},
            'col2': {'null_count': 1, 'null_percentage': 5.0}
        },
        'uniqueness': {
            'col1': {'unique_count': 5, 'unique_percentage': 100},
            'col2': {'unique_count': 4, 'unique_percentage': 95}
        },
        'type_consistency': {},
        'value_ranges': {
            'col1': {'min': 1, 'max': 5}
        }
    }

def test_export_dataset_csv(sample_dataframe):
    """Test CSV export functionality"""
    content, filename, mime_type = export_dataset(sample_dataframe, 'csv')
    
    assert isinstance(content, bytes)
    assert filename.endswith('.csv')
    assert mime_type == 'text/csv'
    
    # Verify content can be read back as CSV
    df_result = pd.read_csv(io.BytesIO(content))
    assert df_result.shape == sample_dataframe.shape

def test_export_dataset_excel(sample_dataframe):
    """Test Excel export functionality"""
    content, filename, mime_type = export_dataset(sample_dataframe, 'excel')
    
    assert isinstance(content, bytes)
    assert filename.endswith('.xlsx')
    assert mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    
    # Verify content can be read back as Excel
    df_result = pd.read_excel(io.BytesIO(content))
    assert df_result.shape == sample_dataframe.shape

def test_export_ml_artifacts(sample_ml_model):
    """Test ML model export functionality"""
    content, filename, mime_type = export_ml_artifacts(sample_ml_model, 'test_model')
    
    assert isinstance(content, bytes)
    assert filename.endswith('.joblib')
    assert mime_type == 'application/octet-stream'
    
    # Verify model can be loaded back
    loaded_model = joblib.load(io.BytesIO(content))
    assert isinstance(loaded_model, type(sample_ml_model))

def test_export_quality_report(sample_quality_metrics):
    """Test quality report export functionality"""
    # Test markdown format
    content, filename, mime_type = export_quality_report(sample_quality_metrics, 'markdown')
    assert isinstance(content, bytes)
    assert filename.endswith('.md')
    assert mime_type == 'text/markdown'
    
    # Test JSON format
    content, filename, mime_type = export_quality_report(sample_quality_metrics, 'json')
    assert isinstance(content, bytes)
    assert filename.endswith('.json')
    assert mime_type == 'application/json'
    
    # Verify JSON content
    loaded_metrics = json.loads(content.decode('utf-8'))
    assert loaded_metrics == sample_quality_metrics


def test_export_dataset_invalid_format(sample_dataframe):
    """Test error handling for invalid export formats"""
    with pytest.raises(RuntimeError):
        export_dataset(sample_dataframe, 'invalid')

def test_export_dataset_empty():
    """Test error handling for empty DataFrame"""
    empty_df = pd.DataFrame()
    with pytest.raises(RuntimeError):
        export_dataset(empty_df, 'csv')

def test_large_dataset_export(sample_dataframe):
    """Test export of larger datasets"""
    # Create a larger DataFrame
    large_df = pd.concat([sample_dataframe] * 1000, ignore_index=True)
    content, filename, mime_type = export_dataset(large_df, 'csv')
    
    assert isinstance(content, bytes)
    assert len(content) > 0