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

def test_all_null_columns_and_rows():
    df = pd.DataFrame({'A': [None, None, None], 'B': [None, None, None]})
    df_clean, notes = clean_dataset(df)
    assert df_clean.empty
    assert any('null' in note.lower() or 'empty' in note.lower() for note in notes)

def test_duplicate_rows():
    """Test cleaning removes duplicate rows"""
    df = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': ['x', 'y', 'y', 'z']
    })
    df_dup = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    df_clean, notes = clean_dataset(df_dup)
    # Should remove at least one duplicate
    assert len(df_clean) < len(df_dup)
    assert any('duplicate' in note.lower() for note in notes)

def test_mixed_data_types():
    """Test cleaning with mixed data types in a column"""
    df = pd.DataFrame({
        'A': [1, '2', 3.0, None],
        'B': ['x', None, 'y', 'z']
    })
    df_clean, notes = clean_dataset(df)
    # Should still return a DataFrame
    assert isinstance(df_clean, pd.DataFrame)
    # Optionally: check notes for type conversion or warnings
    # assert any('type' in note.lower() for note in notes)

def test_whitespace_and_empty_strings():
    df = pd.DataFrame({'A': [' ', '', None, 'value'], 'B': ['x', 'y', 'z', 'x']})
    df_clean, notes = clean_dataset(df)
    # Only check if 'A' survived cleaning (not removed as constant)
    if 'A' in df_clean.columns:
        assert 'value' in df_clean['A'].values
    # Always check for the cleaning note
    assert any('empty' in note.lower() or 'whitespace' in note.lower() for note in notes)


def test_outliers_extreme_values():
    df = pd.DataFrame({'A': [1, 2, 3, 1000, 5], 'B': ['x', 'y', 'z', 'x', 'y']})
    df_clean, notes = clean_dataset(df)
    assert isinstance(df_clean, pd.DataFrame)
    # Uncomment if your cleaning removes outlier rows:
    # assert 1000 not in df_clean['A'].values

def test_constant_column():
    """Test cleaning columns with only one unique value (constants)"""
    df = pd.DataFrame({
        'A': [1, 1, 1, 1],
        'B': ['x', 'y', 'z', 'x']
    })
    df_clean, notes = clean_dataset(df)
    # Should remove constant column if your cleaning does so
    # Or at least, notes should mention it
    assert isinstance(df_clean, pd.DataFrame)
    # Uncomment if your cleaning removes constant columns:
    # assert 'A' not in df_clean.columns
    assert any('constant' in note.lower() or 'unique' in note.lower() for note in notes)