import streamlit as st
from typing import Any, Dict, Optional
import pandas as pd
import hashlib
import json
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def generate_cache_key(data: Any) -> str:
    """Generate a unique cache key based on input data"""
    if isinstance(data, pd.DataFrame):
        # For DataFrames, use shape and column info
        key = f"{data.shape}_{list(data.columns)}_{data.index.size}"
    else:
        # For other data types, use string representation
        key = str(data)
    return hashlib.md5(key.encode()).hexdigest()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cache_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cache DataFrame for faster access"""
    return df

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cache_analysis_results(analysis_type: str, results: Dict) -> Dict:
    """Cache analysis results"""
    return results

@st.cache_resource
def get_cached_state() -> Dict:
    """Get cached application state"""
    if 'cached_state' not in st.session_state:
        st.session_state.cached_state = {}
    return st.session_state.cached_state

def clear_cache(cache_type: Optional[str] = None):
    """Clear specific or all cache"""
    try:
        if cache_type == 'dataframe':
            cache_dataframe.clear()
            logger.info("DataFrame cache cleared")
        elif cache_type == 'analysis':
            cache_analysis_results.clear()
            logger.info("Analysis results cache cleared")
        elif cache_type == 'state':
            if 'cached_state' in st.session_state:
                del st.session_state.cached_state
                logger.info("State cache cleared")
        else:
            st.cache_data.clear()
            st.cache_resource.clear()
            logger.info("All cache cleared")
    except Exception as e:
        logger.error(f"Cache clearing failed: {str(e)}")