import pytest
import pandas as pd
import numpy as np
from utils.data_quality import assess_data_quality, generate_quality_report

@pytest.fixture
def sample_mixed_df():
    """Fixture providing DataFrame with various data quality issues"""
    return pd.DataFrame({
        'numeric_clean': [1, 2, 3, 4, 5],
        'numeric_missing': [1.5, None, 3.5, None, 5.5],
        'numeric_outliers': [10, 12, 1000, 15, 11],
        'categorical': ['A', 'B', 'A', 'B', None],
        'text_mixed': ['Test', 'test', None, 'TEST', 'test'],
        'dates': pd.date_range('2024-01-01', periods=5),
        'constant': [1, 1, 1, 1, 1]
    })

@pytest.fixture
def sample_dirty_df():
    """Fixture with severe data quality issues"""
    df = pd.DataFrame({
        'mixed_types': [1, 'two', 3.0, None, 'five'],
        'duplicates': [1, 1, 2, 2, 2],
        'all_null': [None] * 5,
        'sparse': [None, None, None, 1, None]
    })
    # Add duplicate rows
    return pd.concat([df, df.iloc[0:2]], ignore_index=True)

def test_completeness_metrics(sample_mixed_df):
    """Test completeness calculation"""
    metrics = assess_data_quality(sample_mixed_df)
    completeness = metrics['completeness']
    
    assert 'overall_completeness' in completeness
    assert completeness['overall_completeness'] > 0
    assert completeness['overall_completeness'] <= 100
    
    # Check specific columns
    missing_counts = completeness['missing_counts']
    assert missing_counts['numeric_clean'] == 0
    assert missing_counts['numeric_missing'] == 2
    assert missing_counts['categorical'] == 1

def test_uniqueness_detection(sample_dirty_df):
    """Test uniqueness and duplicate detection"""
    metrics = assess_data_quality(sample_dirty_df)
    uniqueness = metrics['uniqueness']
    
    assert uniqueness['duplicate_rows'] == 2  # Two duplicated rows
    assert 'unique_percentages' in uniqueness
    
    # Check duplicate values in specific column
    unique_vals = uniqueness['unique_values_per_column']['duplicates']
    assert unique_vals == 2  # Only two unique values: 1 and 2

def test_type_consistency(sample_mixed_df, sample_dirty_df):
    """Test data type consistency checks"""
    # Clean data
    clean_metrics = assess_data_quality(sample_mixed_df)
    assert len(clean_metrics['type_consistency']['mixed_types']) == 0
    
    # Dirty data
    dirty_metrics = assess_data_quality(sample_dirty_df)
    mixed_types = dirty_metrics['type_consistency']['mixed_types']
    assert 'mixed_types' in mixed_types  # Column with mixed types
    assert len(mixed_types['mixed_types']) > 1  # Multiple types detected

def test_outlier_detection(sample_mixed_df):
    """Test outlier detection functionality"""
    metrics = assess_data_quality(sample_mixed_df)
    ranges = metrics['value_ranges']
    
    # Check numeric columns
    outliers = ranges['numeric_outliers']['outliers']
    assert outliers['count'] == 1  # One outlier (1000)
    assert outliers['percentage'] > 0
    
    # Check clean numeric column
    clean_outliers = ranges['numeric_clean']['outliers']
    assert clean_outliers['count'] == 0

@pytest.mark.parametrize("invalid_input", [
    pd.DataFrame(),  # Empty DataFrame
    pd.DataFrame({'all_null': [None, None]}),  # All null
    None,  # None input
    "not a dataframe"  # Wrong type
])
def test_invalid_inputs(invalid_input):
    """Test handling of invalid inputs"""
    with pytest.raises((ValueError, TypeError)):
        assess_data_quality(invalid_input)

def test_quality_report_generation(sample_mixed_df):
    """Test quality report generation"""
    report = generate_quality_report(sample_mixed_df)
    
    assert isinstance(report, str)
    assert "Data Quality Assessment Report" in report
    assert "Completeness" in report
    assert "Uniqueness" in report
    assert "Data Type Consistency" in report

def test_large_dataset_performance():
    """Test performance with larger datasets"""
    large_df = pd.DataFrame({
        'numeric': np.random.randn(10000),
        'categorical': np.random.choice(['A', 'B', 'C'], 10000),
        'missing': np.random.choice([1, None], 10000)
    })
    
    import time
    start_time = time.time()
    metrics = assess_data_quality(large_df)
    execution_time = time.time() - start_time
    
    assert execution_time < 5  # Should complete within 5 seconds
    assert isinstance(metrics, dict)

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    edge_cases = pd.DataFrame({
        'infinity': [np.inf, -np.inf, 1, 2],
        'tiny_values': [1e-10, 1e-9, 1e-8],
        'huge_values': [1e10, 1e9, 1e8],
        'special_chars': ['%#@', 'normal', '12!@'],
        'boolean': [True, False, None, True]
    })
    
    metrics = assess_data_quality(edge_cases)
    assert isinstance(metrics, dict)
    assert metrics['completeness']['overall_completeness'] > 0