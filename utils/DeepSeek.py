import os
from typing import Dict, Any, Optional
import requests
import logging
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class DeepSeekChat:
    def __init__(self):
        """Initialize DeepSeek chat with API token"""
        self.api_token = os.getenv('DEEPSEEK_API_TOKEN')
        if not self.api_token:
            logger.error("DEEPSEEK_API_TOKEN not found in environment variables")
            raise ValueError("DEEPSEEK_API_TOKEN not found in environment variables")
        
        if not self.api_token.startswith('sk-or-v1-'):
            logger.error("Invalid DeepSeek API token format")
            raise ValueError("Invalid DeepSeek API token format. Should start with 'sk-or-v1-'")
        
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # verify API token on initialization
        self._verify_token()

    def _verify_token(self) -> None:
        """Verify API token is valid by making a test request"""
        try:
            test_prompt = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=test_prompt,
                timeout=5
            )
            response.raise_for_status()
        except RequestException as e:
            if e.response is not None and e.response.status_code == 401:
                logger.error("Invalid API token: Authentication failed")
                raise ValueError("Invalid API token: Authentication failed")
            logger.error(f"API verification failed: {str(e)}")
            raise RuntimeError(f"API verification failed: {str(e)}")
    

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

    def get_response(self, prompt: dict) -> Optional[str]:
        """Get response from DeepSeek API with enhanced error handling"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=prompt,
                timeout=30
            )
            response.raise_for_status()
            
            if 'error' in response.json():
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                logger.error(f"API Error: {error_msg}")
                return None
                
            return response.json()['choices'][0]['message']['content']
            
        except RequestException as e:
            if e.response is not None and e.response.status_code == 401:
                logger.error("Authentication failed: Please check your API token")
                return "Authentication failed: Please check your API token"
            logger.error(f"API request failed: {str(e)}")
            return f"API request failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"