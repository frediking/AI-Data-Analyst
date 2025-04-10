import os
import replicate
import logging
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

class ReplicateChat:
    def __init__(self):
        """Initialize Replicate chat with API token"""
        self.api_token = os.getenv('REPLICATE_API_TOKEN')
        if not self.api_token:
            logger.error("REPLICATE_API_TOKEN not found in environment variables")
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
        
        # Updated to a more reliable model version
        self.model_ref = "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf"
        self.max_desc_length = 100
        
        # Ensure token is set for replicate package
        replicate.Client(api_token=self.api_token)

    def _truncate_descriptions(self, descriptions: Dict) -> Dict:
        """Truncate column descriptions to manage token length"""
        truncated = {}
        for col, desc in descriptions.items():
            if len(str(desc)) > self.max_desc_length:
                truncated[col] = str(desc)[:self.max_desc_length] + "..."
            else:
                truncated[col] = str(desc)
        return truncated

    def chat_with_data(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate response about the data"""
        try:
            prompt = self._generate_prompt(df_info, question)
            response = self._get_response(prompt)
            
            if not response:
                return "Sorry, I couldn't generate a response. Please try again."
                
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Sorry, I couldn't process that request: {str(e)}"

    def _generate_prompt(self, df_info: Dict[str, Any], question: str) -> str:
        """Generate context-aware prompt"""
        try:
            descriptions = self._truncate_descriptions(df_info.get('descriptions', {}))
            
            # Simplified prompt
            return f"""Based on this dataset:
Rows: {df_info.get('rows', 'N/A')}
Columns: {', '.join(df_info.get('columns', [])[:5])}
Stats: {str(descriptions)[:200]}

Question: {question}

Provide a brief analysis with:
1. Direct answer
2. Key insights
3. Limitations"""
            
        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate prompt: {str(e)}")

    def _get_response(self, prompt: str) -> str:
        """Get response from Replicate API"""
        try:
            # Run model with specific parameters
            output = replicate.run(
                self.model_ref,
                input={
                    "debug": True,
                    "top_k": 50,
                    "top_p": 0.9,
                    "prompt": prompt,
                    "max_length": 500,
                    "temperature": 0.7,
                    "repetition_penalty": 1.1,
                    "system_prompt": "You are a helpful data analysis assistant."
                }
            )
            
            # Handle streaming response
            response_text = ""
            for text in output:
                if text is not None:
                    response_text += str(text)
                    
            if not response_text.strip():
                raise ValueError("Empty response from API")
                
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")