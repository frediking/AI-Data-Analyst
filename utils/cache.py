import streamlit as st
from typing import Any, Dict, Optional
import pandas as pd
import hashlib
import logging

logger = logging.getLogger(__name__)

# In-memory cache dictionaries for demonstration/testing
_dataframe_cache = {}
_analysis_cache = {}

def generate_cache_key(data: Any) -> str:
    """Generate a unique cache key based on input data"""
    if isinstance(data, pd.DataFrame):
        key = f"{data.shape}_{list(data.columns)}_{data.index.size}"
    else:
        key = str(data)
    return hashlib.md5(key.encode()).hexdigest()

def cache_dataframe(df: pd.DataFrame, key: str = None) -> pd.DataFrame:
    """Cache DataFrame for faster access."""
    if key is None:
        key = generate_cache_key(df)
    _dataframe_cache[key] = df
    return df

def cache_analysis_results(analysis_type: str, results: Dict, key: str = None) -> Dict:
    """Cache analysis results."""
    if key is None:
        key = f"{analysis_type}_{hash(str(results))}"
    _analysis_cache[key] = results
    return results

def get_cached_dataframe(key: str) -> Optional[pd.DataFrame]:
    """Retrieve cached DataFrame by key."""
    return _dataframe_cache.get(key)

def get_cached_analysis(key: str) -> Optional[Dict]:
    """Retrieve cached analysis results by key."""
    return _analysis_cache.get(key)

def clear_cache():
    """Clear all caches."""
    _dataframe_cache.clear()
    _analysis_cache.clear()