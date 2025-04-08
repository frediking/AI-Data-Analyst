import pytest
import pandas as pd
import os
import shutil
from utils.cache import (
    cache_dataframe,
    cache_analysis_results,
    clear_cache,
    get_cached_dataframe,
    get_cached_analysis
)

@pytest.fixture(scope="function")
def setup_cache_dir():
    """Setup and cleanup temporary cache directory"""
    cache_dir = ".streamlit/cache"
    os.makedirs(cache_dir, exist_ok=True)
    yield cache_dir
    # Cleanup after tests
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })

@pytest.fixture
def sample_analysis():
    """Create sample analysis results for testing"""
    return {
        'summary': 'Test analysis',
        'metrics': {'mean': 2.0, 'std': 1.0},
        'timestamp': '2024-04-08'
    }

def test_cache_dataframe(setup_cache_dir, sample_dataframe):
    """Test DataFrame caching functionality"""
    # Test caching
    cache_key = cache_dataframe(sample_dataframe, 'test_df')
    assert cache_key is not None
    
    # Test retrieval
    cached_df = get_cached_dataframe(cache_key)
    assert cached_df is not None
    assert cached_df.equals(sample_dataframe)

def test_cache_analysis(setup_cache_dir, sample_analysis):
    """Test analysis results caching functionality"""
    # Test caching
    cache_key = cache_analysis_results(sample_analysis, 'test_analysis')
    assert cache_key is not None
    
    # Test retrieval
    cached_analysis = get_cached_analysis(cache_key)
    assert cached_analysis is not None
    assert cached_analysis == sample_analysis

def test_clear_cache(setup_cache_dir, sample_dataframe, sample_analysis):
    """Test cache clearing functionality"""
    # Cache some data
    df_key = cache_dataframe(sample_dataframe, 'test_df')
    analysis_key = cache_analysis_results(sample_analysis, 'test_analysis')
    
    # Clear specific cache
    clear_cache('dataframe')
    assert get_cached_dataframe(df_key) is None
    assert get_cached_analysis(analysis_key) is not None
    
    # Clear all cache
    clear_cache()
    assert get_cached_dataframe(df_key) is None
    assert get_cached_analysis(analysis_key) is None

@pytest.mark.parametrize("invalid_key", [
    None,
    "",
    123,
    "nonexistent_key"
])
def test_cache_retrieval_invalid_keys(setup_cache_dir, invalid_key):
    """Test cache retrieval with invalid keys"""
    assert get_cached_dataframe(invalid_key) is None
    assert get_cached_analysis(invalid_key) is None

def test_cache_large_dataframe(setup_cache_dir):
    """Test caching performance with large DataFrame"""
    large_df = pd.DataFrame({
        'A': range(10000),
        'B': ['test'] * 10000
    })
    
    cache_key = cache_dataframe(large_df, 'large_df')
    cached_df = get_cached_dataframe(cache_key)
    assert cached_df is not None
    assert cached_df.equals(large_df)

def test_concurrent_cache_access(setup_cache_dir, sample_dataframe):
    """Test concurrent cache access"""
    import threading
    
    def cache_operation():
        cache_key = cache_dataframe(sample_dataframe, f'test_df_{threading.get_ident()}')
        assert get_cached_dataframe(cache_key) is not None
    
    threads = [threading.Thread(target=cache_operation) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()