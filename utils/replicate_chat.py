import os
import replicate
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReplicateChat:
    def __init__(self):
        """Initialize Replicate chat with API token"""
        self.api_token = os.getenv('REPLICATE_API_TOKEN')
        if not self.api_token:
            logger.error("REPLICATE_API_TOKEN not found in environment variables")
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
        
        # Define model reference
        self.model_ref = "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1"

    def chat_with_data(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate response about the data"""
        try:
            prompt = self._generate_prompt(df_info, question)
            return self._get_response(prompt)
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Sorry, I couldn't process that request: {str(e)}"

    def _generate_prompt(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate context-aware prompt"""
        return f"""Dataset Info:
- Rows: {df_info.get('rows', 'N/A')}
- Columns: {df_info.get('columns', [])}
- Data types: {df_info.get('dtypes', {})}
- Column descriptions: {df_info.get('descriptions', {})}

Question: {question}

Please provide:
1. Direct answer to the question
2. Key insights relevant to the question
3. Any important caveats or limitations"""

    def _get_response(self, prompt: str) -> str:
        """Get response from Replicate API"""
        try:
            # Use the model reference when calling replicate.run
            output = replicate.run(
                self.model_ref,
                input={
                    "prompt": prompt,
                    "temperature": 0.1,
                    "max_new_tokens": 500,
                    "system_prompt": "You are an AI data analyst assistant."
                }
            )
            return "".join(output).strip()
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")