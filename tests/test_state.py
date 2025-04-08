import pytest
import streamlit as st
import pandas as pd
from utils.state import initialize_session_state, update_state_after_cleaning

@pytest.fixture
def mock_session_state(monkeypatch):
    """Mock Streamlit session state"""
    class MockState(dict):
        def __init__(self):
            super().__init__()
            self._is_running = True

    mock_state = MockState()
    monkeypatch.setattr(st, "session_state", mock_state)
    return mock_state

def test_initialize_session_state(mock_session_state):
    """Test session state initialization"""
    initialize_session_state()
    
    # Check if all required keys are present
    assert 'data' in st.session_state
    assert 'cleaning_history' in st.session_state
    assert 'analysis_results' in st.session_state
    assert 'ml_state' in st.session_state
    
    # Check initial values
    assert st.session_state.data is None
    assert isinstance(st.session_state.cleaning_history, list)
    assert len(st.session_state.cleaning_history) == 0
    assert isinstance(st.session_state.analysis_results, dict)

def test_update_state_after_cleaning(mock_session_state):
    """Test state updates after cleaning operations"""
    # Initialize state first
    initialize_session_state()
    
    # Create test data
    df_clean = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    cleaning_notes = [
        "Removed duplicates",
        "Standardized column names"
    ]
    
    # Update state
    update_state_after_cleaning(df_clean, cleaning_notes)
    
    # Verify updates
    assert st.session_state.data is not None
    assert st.session_state.data.equals(df_clean)
    assert len(st.session_state.cleaning_history) == len(cleaning_notes)
    assert all(note in st.session_state.cleaning_history for note in cleaning_notes)

def test_ml_state_initialization(mock_session_state):
    """Test ML state initialization"""
    initialize_session_state()
    
    # Check ML state structure
    assert 'ml_state' in st.session_state
    assert isinstance(st.session_state.ml_state, dict)
    assert 'current_model' in st.session_state.ml_state
    assert 'target_column' in st.session_state.ml_state
    assert 'feature_importance' in st.session_state.ml_state

@pytest.mark.parametrize("invalid_df", [
    None,
    "not a dataframe",
    123
])
def test_update_state_invalid_input(mock_session_state, invalid_df):
    """Test error handling for invalid inputs"""
    initialize_session_state()
    
    with pytest.raises(ValueError):
        update_state_after_cleaning(invalid_df, ["test note"])