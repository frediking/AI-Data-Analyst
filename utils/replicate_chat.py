import os
import replicate
import streamlit as st
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ReplicateChat:
    def __init__(self):
        """Initialize Replicate chat with API token"""
        self.api_token = os.getenv('REPLICATE_API_TOKEN')
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
        
        # Llama 2 model settings
        self.model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
        self.temperature = 0.7
        self.max_length = 500
        self.top_p = 0.9

    def generate_prompt(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate context-aware prompt for Llama 2"""
        return f"""You are an AI data analyst assistant. Analyze this data and answer the question.

Dataset Info:
- Rows: {df_info.get('rows', 'N/A')}
- Columns: {df_info.get('columns', [])}
- Data types: {df_info.get('dtypes', {})}
- Column descriptions: {df_info.get('descriptions', {})}

User Question: {question}

Provide a clear, concise answer based on the data context provided."""

    def get_response(self, prompt: str) -> str:
        """Get response from Llama 2 model"""
        try:
            # Initialize Replicate client
            output = replicate.run(
                self.model,
                input={
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_length": self.max_length,
                    "top_p": self.top_p
                }
            )
            
            # Concatenate streaming output
            response = ""
            for item in output:
                response += item
                
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    @st.cache_data(ttl=3600)
    def chat_with_data(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate response about the data"""
        try:
            prompt = self.generate_prompt(df_info, question)
            response = self.get_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Sorry, I couldn't process that request: {str(e)}"