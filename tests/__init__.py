import pytest
import pandas as pd
from utils.data_quality import assess_data_quality

def test_data_quality_basic():
    """Test basic functionality of data quality assessment"""
    # Create sample DataFrame
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': ['x', 'y', 'z', 'x']
    })
    
    # Run quality assessment
    metrics = assess_data_quality(df)
    
    # Basic assertions
    assert isinstance(metrics, dict)
    assert 'completeness' in metrics
    assert 'uniqueness' in metrics