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
        
        # Use the latest stable version of the model
        self.model_ref = "meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0"
        self.max_desc_length = 100
        
        # Set token in environment
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        replicate.Client(api_token=self.api_token)  # Initialize client
        
        # Verify token is valid
        self._verify_token()
        
    def _verify_token(self) -> None:
        """Verify the API token is valid"""
        try:
            # Simple test run with streaming
            test_response = ""
            for event in replicate.stream(
                self.model_ref,
                input={
                    "prompt": "test",
                    "max_tokens": 10,
                    "temperature": 0.1,
                    "system_prompt": "You are a helpful assistant"
                }
            ):
                if event is not None:
                    test_response += str(event)
                    break  # Just need first token to verify
                
            if not test_response:
                raise ValueError("No response received during verification")
            
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

    def chat_with_data(self, df, prompt):
        """Enhanced version that understands data context"""
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
           raise ValueError("Prompt is required for the chatbot API and cannot be empty.")

        system_prompt = """You're a data analyst assistant. 
        When responding, consider:
        1. The active tab user is viewing
        2. Recent operations they performed
        3. Currently open visualizations
        """
        response = replicate.run(
            self.model_ref,
            input={"prompt": prompt, "system_prompt": system_prompt}        
        )

        return "".join([msg["content"] for msg in response])

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
        """Get response from Replicate API using streaming"""
        try:
            # Create complete prompt with system message
            response_text = ""
            for event in replicate.stream(
                self.model_ref,
                input={
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "system_prompt": "You are a helpful data analysis assistant",
                    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                    "length_penalty": 1,
                    "presence_penalty": 0,
                    "log_performance_metrics": False
                }
            ):
                if event is not None:
                    response_text += str(event)
                
            if not response_text.strip():
                raise ValueError("Empty response from API")
            
            return response_text.strip()
        
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")