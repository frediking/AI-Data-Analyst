import pytest
from utils.replicate_chat import ReplicateChat
import os

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv('REPLICATE_API_TOKEN', 'test_token')

def test_chat_initialization(mock_env):
    """Test chat initialization with mock token"""
    chat = ReplicateChat()
    assert chat is not None
    assert chat.api_token == 'test_token'

@pytest.mark.parametrize("df_info,question", [
    (
        {
            'rows': 5,
            'columns': ['A', 'B'],
            'dtypes': {'A': 'int64', 'B': 'object'},
            'descriptions': {'A': 'numeric column', 'B': 'text column'}
        },
        "What are the columns in this dataset?"
    )
])
def test_prompt_generation(mock_env, df_info, question):
    """Test prompt generation with sample data"""
    chat = ReplicateChat()
    prompt = chat.generate_prompt(df_info, question)
    assert isinstance(prompt, str)
    assert "Dataset Info" in prompt
    assert question in prompt