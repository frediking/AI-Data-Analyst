import os
from typing import Dict, Any
import requests
import logging

logger = logging.getLogger(__name__)

class DeepSeekChat:
    def __init__(self):
        """Initialize DeepSeek chat with API token"""
        self.api_token = os.getenv('DEEPSEEK_API_TOKEN')
        if not self.api_token:
            raise ValueError("DEEPSEEK_API_TOKEN not found in environment variables")
        
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def generate_prompt(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate context-aware prompt"""
        return {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an AI data analyst assistant."},
                {"role": "user", "content": f"""
                Dataset Info:
                - Rows: {df_info.get('rows', 'N/A')}
                - Columns: {df_info.get('columns', [])}
                - Data types: {df_info.get('dtypes', {})}
                - Column descriptions: {df_info.get('descriptions', {})}

                Question: {question}
                """}
            ],
            "temperature": 0.7
        }

    def get_response(self, prompt: dict) -> str:
        """Get response from DeepSeek API"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=prompt
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def chat_with_data(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate response about the data"""
        try:
            prompt = self.generate_prompt(df_info, question)
            response = self.get_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Sorry, I couldn't process that request: {str(e)}"