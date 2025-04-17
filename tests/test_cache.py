import pytest
import pandas as pd
from utils.cache import (
    cache_dataframe,
    cache_analysis_results,
    get_cached_dataframe,
    get_cached_analysis,
    clear_cache,
    generate_cache_key,
)

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})

def test_cache_and_retrieve_dataframe(sample_dataframe):
    clear_cache()
    key = generate_cache_key(sample_dataframe)
    cache_dataframe(sample_dataframe, key)
    cached = get_cached_dataframe(key)
    assert cached.equals(sample_dataframe)

def test_cache_and_retrieve_analysis():
    clear_cache()
    key = "test_analysis"
    results = {"foo": 123, "bar": 456}
    cache_analysis_results("test", results, key)
    cached = get_cached_analysis(key)
    assert cached == results

def test_clear_cache(sample_dataframe):
    key = generate_cache_key(sample_dataframe)
    cache_dataframe(sample_dataframe, key)
    clear_cache()
    assert get_cached_dataframe(key) is None