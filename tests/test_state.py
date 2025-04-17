import pytest
import streamlit as st
import pandas as pd
from utils.state import initialize_session_state, update_state_after_cleaning

@pytest.fixture
def mock_session_state(monkeypatch):
    """Mock Streamlit session state with both dict and attribute access."""
    class MockState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value
    mock_state = MockState()
    monkeypatch.setattr(st, "session_state", mock_state)
    return mock_state

def test_initialize_session_state(mock_session_state):
    """Test session state initialization"""
    initialize_session_state()
    
    # Check if all required keys are present
    assert 'data_state' in st.session_state
    assert 'app_state' in st.session_state
    assert 'viz_settings' in st.session_state
    assert 'dashboard_state' in st.session_state
    assert 'version_state' in st.session_state
    assert 'ml_state' in st.session_state
    assert 'chat_state' in st.session_state
    
    # Check initial data_state values
    assert st.session_state.data_state == {
        'cleaned': False,
        'cleaning_notes': [],
        'original_df': None,
        'current_df': None,
        'last_operation': None,
        'can_undo': False,
        'processing_history': []
    }
    
    # Check app_state keys (version and last_updated)
    assert 'version' in st.session_state.app_state
    assert 'last_updated' in st.session_state.app_state
    
    # Check viz_settings
    assert st.session_state.viz_settings == {
        'last_chart_type': None,
        'custom_colors': None
    }
    
    # Check dashboard_state
    assert st.session_state.dashboard_state == {
        'last_chart_type': None,
        'selected_columns': [],
        'chart_settings': {}
    }
    
    # Check version_state
    assert st.session_state.version_state == {
        'current_version': None,
        'version_history': [],
        'can_restore': False,
        'last_saved_hash': None
    }
    
    # Check ml_state
    assert st.session_state.ml_state == {
        'current_model': None,
        'target_column': None,
        'last_prediction': None,
        'feature_importance': None,
        'model_performance': None,
        'task_type': None
    }
    
    # Check chat_state
    assert st.session_state.chat_state == {
        'history': [],
        'model_loaded': False,
        'last_prompt': None,
        'context': None,
        'messages': []
    }
    
    # Check messages and chat
    assert st.session_state.messages == []
    assert st.session_state.chat is None

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
    assert st.session_state.data_state['current_df'] is not None
    assert st.session_state.data_state['current_df'].equals(df_clean)
    assert len(st.session_state.data_state['cleaning_notes']) == len(cleaning_notes)
    assert all(note in st.session_state.data_state['cleaning_notes'] for note in cleaning_notes)

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
    
    with pytest.raises(TypeError):
        update_state_after_cleaning(invalid_df, ["test note"])