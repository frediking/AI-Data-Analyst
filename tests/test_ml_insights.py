import pytest
import pandas as pd
from utils.ml_insights import MLInsights

def test_ml_insights_initialization():
    """Test MLInsights class initialization"""
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4],
        'categorical': ['A', 'B', 'A', 'B']
    })
    
    insights = MLInsights(df)
    assert insights is not None
    assert len(insights.numeric_cols) == 1
    assert "numeric" in insights.numeric_cols
    assert len(insights.categorical_cols) == 1
    assert "categorical" in insights.categorical_cols

    
def test_ml_insights_empty_df():
    with pytest.raises(ValueError):
        MLInsights(pd.DataFrame())