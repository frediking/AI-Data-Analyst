import pytest
import pandas as pd
from utils.preprocessing import clean_dataset

def test_basic_cleaning():
    """Test basic data cleaning operations"""
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': ['x', 'y', 'z', 'x']
    })
    
    df_clean, notes = clean_dataset(df)
    assert df_clean is not None
    assert isinstance(notes, list)
    assert len(df_clean) <= len(df)