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
        
        # Use a more stable model reference
        self.model_ref = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
        self.max_desc_length = 100
        
        # Set environment variable
        os.environ["REPLICATE_API_TOKEN"] = self.api_token

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
            
            # More concise prompt
            return f"""Analyze this dataset:
- Rows: {df_info.get('rows', 'N/A')}
- Columns: {', '.join(df_info.get('columns', [])[:5])}... 
- Key stats: {str(descriptions)[:200]}...

Question: {question}

Provide a concise response with:
1. Direct answer
2. Key insights
3. Limitations"""
            
        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate prompt: {str(e)}")

    def _get_response(self, prompt: str) -> str:
        """Get response from Replicate API"""
        try:
            # Run the model with retries
            for attempt in range(3):
                try:
                    output = replicate.run(
                        self.model_ref,
                        input={
                            "prompt": prompt,
                            "temperature": 0.1,
                            "max_tokens": 500,
                            "top_p": 0.9,
                            "system_prompt": "You are a helpful data analysis assistant."
                        }
                    )
                    
                    # Collect streaming output
                    response = ""
                    for item in output:
                        response += item
                    
                    return response.strip()
                    
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise e
                    time.sleep(1)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")