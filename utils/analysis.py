import pandas as pd
from typing import Optional, Dict, Any
from functools import lru_cache
import logging
import os
import requests

logger = logging.getLogger(__name__)

def validate_input(summary: str, prompt: str) -> None:
    """Validate input parameters"""
    if not summary or not isinstance(summary, str):
        raise ValueError("Invalid or empty data summary")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid or empty prompt")

@lru_cache(maxsize=1)
def load_model():
    """Load the DeepSeek model with error handling"""
    try:
        # Check for API token
        if not os.getenv('DEEPSEEK_API_TOKEN'):
            raise ValueError("DEEPSEEK_API_TOKEN not found in environment variables")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to initialize Replicate: {str(e)}")

def prepare_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Prepare additional context about the dataset"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    try:
        context = {
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "column_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        if context["numeric_columns"]:
            context["numeric_stats"] = df[context["numeric_columns"]].describe().to_dict()
            
        return context
    except Exception as e:
        logger.error(f"Context preparation failed: {str(e)}")
        raise ValueError(f"Failed to prepare context: {str(e)}")

def generate_analysis(summary: str, prompt: str, df: Optional[pd.DataFrame] = None) -> str:
    """Generate analysis based on data summary and user prompt"""
    try:
        # Validate inputs first
        validate_input(summary, prompt)
        
        # Load model with connection test
        load_model()
        
        # Prepare context if DataFrame provided
        context = ""
        if df is not None:
            try:
                context_dict = prepare_context(df)
                context = "\n".join([f"- {k}: {v}" for k, v in context_dict.items()])
            except Exception as e:
                logger.warning(f"Context preparation failed: {str(e)}")
                context = "No additional context available"
        
        
        # Create messages for DeepSeek API
        messages = [
            {"role": "system", "content": "You are an expert data analyst. Provide clear, concise, and accurate analysis."},
            {"role": "user", "content": f"""

        

Dataset Summary:
{summary}

Additional Context:
{context}

Question: {prompt}

Please provide:
1. Direct answer to the question
2. Key insights relevant to the question
3. Any important caveats or limitations"""}
        ]

        # Use DeepSeek API
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_TOKEN')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 500
            },
            timeout=30
        )
        

        # Check for API error response
        if 'error' in response.json():
            raise RuntimeError(f"API Error: {response.json()['error']}")
            
        return response.json()['choices'][0]['message']['content'].strip()

    
    except Exception as e:
        logger.error(f"Analysis generation failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Analysis generation failed: {str(e)}")