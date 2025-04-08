import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

__version__ = "1.0.0"  # Define here instead of importing from app to avoid circular imports

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'data_state' not in st.session_state:
        st.session_state.data_state = {
            'cleaned': False,
            'cleaning_notes': [],
            'original_df': None,
            'current_df': None,
            'last_operation': None,
            'can_undo': False,
            'processing_history': []
        }
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            'version': __version__,
            'last_updated': datetime.now().isoformat(),
        }

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'viz_settings' not in st.session_state:
        st.session_state.viz_settings = {
            'last_chart_type': None,
            'custom_colors': None
        }

    if 'dashboard_state' not in st.session_state:
        st.session_state.dashboard_state = {
        'last_chart_type': None,
        'selected_columns': [],
        'chart_settings': {}
    }
    
        
    if 'version_state' not in st.session_state:
        st.session_state.version_state = {
            'current_version': None,
            'version_history': [],
            'can_restore': False,
            'last_saved_hash': None
        }

    if 'ml_state' not in st.session_state:
        st.session_state.ml_state = {
            'current_model': None,
            'target_column': None,
            'last_prediction': None,
            'feature_importance': None,
            'model_performance': None,
            'task_type': None
        }

    if 'chat_state' not in st.session_state:
        st.session_state.chat_state = {
            'history': [],
            'model_loaded': False,
            'last_prompt': None,
            'context': None,
            'messages': []  # For storing chat messages
        }

    logger.info("Session state initialized successfully")    

def update_state_after_cleaning(df: pd.DataFrame, notes: List[str]) -> None:
    """
    Update session state after cleaning operations
    
    Args:
        df: Cleaned DataFrame
        notes: List of cleaning operation notes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(notes, list):
        raise TypeError("notes must be a list of strings")
        
    st.session_state.data_state['cleaned'] = True
    st.session_state.data_state['current_df'] = df
    st.session_state.data_state['cleaning_notes'].extend(notes)
    st.session_state.data_state['can_undo'] = True
    st.session_state.data_state['last_operation'] = 'cleaning'