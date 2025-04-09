import pytest
import os
from utils.DeepSeek import DeepSeekChat

@pytest.fixture
def mock_env(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv('DEEPSEEK_API_TOKEN', 'test_token')

@pytest.fixture
def sample_data():
    """Sample DataFrame info for testing"""
    return {
        'rows': 5,
        'columns': ['A', 'B'],
        'dtypes': {'A': 'int64', 'B': 'object'},
        'descriptions': {'A': 'Numbers', 'B': 'Text'}
    }

def test_chat_initialization(mock_env):
    """Test chat initialization with API token"""
    chat = DeepSeekChat()
    assert chat.api_token == 'test_token'
    assert 'Authorization' in chat.headers
    assert 'Content-Type' in chat.headers

def test_chat_missing_token():
    """Test initialization without API token"""
    with pytest.raises(ValueError):
        DeepSeekChat()

def test_chat_with_data(mock_env, sample_data):
    """Test basic chat functionality"""
    chat = DeepSeekChat()
    response = chat.chat_with_data(sample_data, "What columns are in the dataset?")
    assert isinstance(response, str)
    assert len(response) > 0