import os
import replicate
import streamlit as st
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ReplicateChat:
    def __init__(self):
        """Initialize Replicate chat with API token"""
        self.api_token = os.getenv('REPLICATE_API_TOKEN')
        if not self.api_token:
            logger.error("REPLICATE_API_TOKEN not found in environment variables")
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")

    @st.cache_data(ttl=3600)
    def _get_cached_response(
        _self,  # Add underscore to prevent hashing
        df_info: Dict[str, Any],
        question: str
    ) -> str:
        """Get cached response from Replicate API"""
        prompt = _self._generate_prompt(df_info, question)
        return _self._get_response(prompt)

    def chat_with_data(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate response about the data"""
        try:
            return self._get_cached_response(df_info, question)
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Sorry, I couldn't process that request: {str(e)}"

    def _generate_prompt(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate context-aware prompt"""
        return {
            "model": "replicate/ollama-mistral-7b:2f3c5fa8ac882bcbbef749985e3a0c607ac61e03c6555016c118872ef0c19685",
            "input": {
                "prompt": f"""
                Dataset Info:
                - Rows: {df_info.get('rows', 'N/A')}
                - Columns: {df_info.get('columns', [])}
                - Data types: {df_info.get('dtypes', {})}
                - Column descriptions: {df_info.get('descriptions', {})}

                Question: {question}

                Please provide:
                1. Direct answer to the question
                2. Key insights relevant to the question
                3. Any important caveats or limitations
                """,
                "system_prompt": "You are an AI data analyst assistant.",
                "temperature": 0.7,
                "max_length": 500
            }
        }

    def _get_response(self, prompt: Dict) -> str:
        """Get response from Replicate API"""
        try:
            output = replicate.run(**prompt)
            return "".join(output).strip()
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")