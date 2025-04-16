import pytest
import pandas as pd
from utils.analysis import generate_analysis_payload, prepare_context

def test_prepare_context():
    """Test context preparation for analysis"""
    # Create test DataFrame
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4],
        'text': ['a', 'b', 'c', 'd'],
        'date': pd.date_range('2024-01-01', periods=4)
    })
    
    context = prepare_context(df)
    
    # Basic assertions for the new context structure
    assert isinstance(context, dict)
    assert 'shape' in context
    assert context['shape'] == (4, 3)
    assert 'columns' in context
    assert set(context['columns']) == {'numeric', 'text', 'date'}
    assert 'sample' in context
    assert isinstance(context['sample'], list)
    assert len(context['sample']) == 3  # head(3)
    
def test_generate_analysis_payload():
    """Test analysis payload generation"""
    summary = "Dataset contains numeric and text columns"
    prompt = "What is the distribution of numeric values?"
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4],
        'text': ['a', 'b', 'c', 'd']
    })
    analysis = generate_analysis_payload(summary, prompt, df)
    assert isinstance(analysis, dict)
    assert "data_summary" in analysis
    assert "user_question" in analysis
    assert "context" in analysis
    assert "metadata" in analysis
    assert analysis["data_summary"] == summary
    assert analysis["user_question"] == prompt
    assert set(analysis["context"]["columns"]) == {'numeric', 'text'}

@pytest.mark.parametrize("invalid_input", [
    None,
    "",
    123,
    ["not a string"]
])
def test_generate_analysis_invalid_inputs(invalid_input):
    with pytest.raises(ValueError):
        generate_analysis_payload(invalid_input, invalid_input)