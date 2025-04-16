"""
Data analysis preparation module - prepares data for AI analysis
"""
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

def validate_input(summary: str, prompt: str) -> None:
    """Validate analysis inputs"""
    if not summary or not isinstance(summary, str):
        raise ValueError("Invalid data summary")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid prompt")

def prepare_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Prepare DataFrame context for analysis"""
    try:
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "sample": df.head(3).to_dict(orient='records')
        }
    except Exception as e:
        logger.warning(f"Context preparation failed: {str(e)}")
        return {"error": str(e)}

def generate_analysis_payload(
    summary: str, 
    prompt: str, 
    df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Prepare analysis payload without API calls
    
    Returns:
        {
            "data_summary": str,
            "user_question": str,
            "context": dict,
            "metadata": {
                "timestamp": str,
                "columns": List[str]
            }
        }
    """
    validate_input(summary, prompt)
    
    context = {}
    if df is not None:
        context = prepare_context(df)
    
    return {
        "data_summary": summary,
        "user_question": prompt,
        "context": context,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "columns": list(df.columns) if df is not None else []
        }
    }