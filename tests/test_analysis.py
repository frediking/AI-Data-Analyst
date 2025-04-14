import pytest
import pandas as pd
from utils.analysis import generate_analysis, prepare_context

def test_prepare_context():
    """Test context preparation for analysis"""
    # Create test DataFrame
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4],
        'text': ['a', 'b', 'c', 'd'],
        'date': pd.date_range('2024-01-01', periods=4)
    })
    
    context = prepare_context(df)
    
    # Basic assertions
    assert isinstance(context, dict)
    assert 'total_rows' in context
    assert 'total_columns' in context
    assert context['total_rows'] == 4
    assert context['total_columns'] == 3
    assert 'numeric_columns' in context
    assert 'numeric' in context['numeric_columns']

def test_generate_analysis():
    """Test analysis generation"""
    # Test inputs
    summary = "Dataset contains numeric and text columns"
    prompt = "What is the distribution of numeric values?"
    
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4],
        'text': ['a', 'b', 'c', 'd']
    })
    
    # Generate analysis
    analysis = generate_analysis(summary, prompt, df)
    
    # Basic assertions
    assert isinstance(analysis, str)
    assert len(analysis) > 0

@pytest.mark.parametrize("invalid_input", [
    None,
    "",
    123,
    ["not a string"]
])
def test_generate_analysis_invalid_inputs(invalid_input):
    from utils.analysis import generate_analysis
    with pytest.raises(RuntimeError):
        generate_analysis(invalid_input, invalid_input)