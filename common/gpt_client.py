"""
GPT Client module for LLM integration.
This module provides a unified interface for interacting with LLMs through API.
"""

from openai import OpenAI

class GPTClient:
    """Client for interacting with GPT and compatible models."""
    
    def __init__(self, key=None, url=None, model="gpt-4"):
        """
        Initialize GPT client.
        
        Args:
            key: API key (will use environment variable if None)
            url: API base URL (will use default if None)
            model: Model name to use for queries
        """
        self.model = model or "gpt-4"
        self.api_key = key or "YOUR_API_KEY"  # Replace with environment variable in production
        self.base_url = url or "https://api.openai.com/v1"
        self.client = self.init_connection()

    def init_connection(self):
        """Initialize API connection."""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def get_response(self, messages, model=None):
        """
        Get response from the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            
        Returns:
            Response content as string, or 'Failed' on error
        """
        # Use specified model or default to instance model
        use_model = model or self.model
        
        # Determine model characteristics
        model_name = use_model.lower()
        is_streaming_preferred = 'qwen' in model_name
        
        # Set temperature based on model
        temperature = 0.2 if 'gpt-4' in model_name else 0.3
            
        try:
            if is_streaming_preferred:
                # Stream response for some models
                stream = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=temperature,
                    stream=True
                )
                
                # Collect streamed response
                collected_content = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        collected_content += chunk.choices[0].delta.content
                
                # Check for error messages
                if "request error" in collected_content.lower() or "sorry" in collected_content.lower():
                    return 'Failed'
                return collected_content
            else:
                # Standard response for other models
                completion = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=temperature
                )
                
                # Check for error messages
                content = completion.choices[0].message.content
                if "request error" in content.lower() or "sorry" in content.lower():
                    return 'Failed'
                return content
                
        except Exception as e:
            print(f"API error: {str(e)}")
            return 'Failed'

    def infer_data(self, data):
        """Simple wrapper for get_response."""
        return self.get_response(data) 