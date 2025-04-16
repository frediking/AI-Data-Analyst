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
    assert 'overall_completeness' in metrics
    assert metrics['overall_completeness'] > 0
    assert metrics['overall_completeness'] <= 100

    # Check null counts per column
    null_counts = {col: info['null_count'] for col, info in metrics['null_summary'].items()}
    assert null_counts['numeric_clean'] == 0
    assert null_counts['numeric_missing'] == 2
    assert null_counts['categorical'] == 1

def test_uniqueness_detection(sample_dirty_df):
    """Test uniqueness and duplicate detection"""
    metrics = assess_data_quality(sample_dirty_df)
    uniqueness = metrics['uniqueness']
    # Check unique count for duplicates column
    assert uniqueness['duplicates']['unique_count'] == 2
    # Check unique count for mixed_types column
    assert uniqueness['mixed_types']['unique_count'] == 4
    # All_null column should have 0 unique values (since all are None)
    assert uniqueness['all_null']['unique_count'] == 0

def test_type_consistency(sample_mixed_df, sample_dirty_df):
    """Test data type consistency checks"""
    metrics_clean = assess_data_quality(sample_mixed_df)
    # No mixed types expected in sample_mixed_df
    assert len(metrics_clean['mixed_types']) == 0

    metrics_dirty = assess_data_quality(sample_dirty_df)
    # mixed_types and sparse have mixed types
    assert 'mixed_types' in metrics_dirty['mixed_types']

def test_outlier_detection(sample_mixed_df):
    """Test outlier detection"""
    metrics = assess_data_quality(sample_mixed_df)
    outliers = metrics['outliers']
    # numeric_outliers should have at least 1 outlier (1000)
    assert outliers['numeric_outliers']['count'] >= 1

@pytest.mark.parametrize("invalid_input", [
    pd.DataFrame(),  # Empty DataFrame
    pd.DataFrame({'all_null': [None, None]}),  # All null
    None,  # None input
    "not a dataframe"  # Wrong type
])
def test_invalid_inputs(invalid_input):
    """Test handling of invalid inputs"""
    import pytest
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        assess_data_quality(invalid_input)

def test_quality_report_generation(sample_mixed_df):
    """Test report generation does not crash and contains key sections"""
    report = generate_quality_report(sample_mixed_df)
    assert "Completeness" in report
    assert "Uniqueness" in report
    assert "Data Type Consistency" in report
    assert "Value Ranges" in report

def test_large_dataset_performance():
    """Test that function works on a large DataFrame (not a speed test)"""
    df = pd.DataFrame({
        'A': np.random.randint(0, 100, size=10000),
        'B': np.random.choice(['x', 'y', 'z'], size=10000),
        'C': np.random.randn(10000)
    })
    metrics = assess_data_quality(df)
    assert 'overall_completeness' in metrics
    assert 'uniqueness' in metrics

def test_edge_cases():
    """Test edge cases like columns of different lengths (should raise)"""
    import pytest
    with pytest.raises(ValueError):
        pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1, 2]
        })