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
        
        # Verify token format
        if not self.api_token.startswith('r8_'):
            logger.error("Invalid Replicate API token format")
            raise ValueError("Invalid Replicate API token format. Should start with 'r8_'")
        
        # Use Mistral 7B - free model
        self.model_ref = "meta/meta-llama-3-8b-instruct"
        self.max_desc_length = 100
        
        # Set token in environment
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        
        # Verify token is valid
        self._verify_token()
        
    def _verify_token(self) -> None:
        """Verify the API token is valid"""
        try:
            # Simple test run
            output = replicate.run(
                self.model_ref,
                input={
                    "prompt": "test",
                    "max_new_tokens": 1
                }
            )
            next(output)  # Test if we can get at least one token
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise ValueError(f"Invalid API token: {str(e)}")

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
                    "top_k": 0,
                    "top_p": 0.95,
                    "prompt": prompt,
                    "max_length": 500,
                    "temperature": 0.7,
                    "length_penalty": 1,
                    "repetition_penalty": 1.1,
                    "stop_sequences": "<|end_of_text|>,<|eot_id|>",
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