import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer

class HFInferenceClient:
    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the HuggingFace Inference client.
        
        Args:
            model_id: The model ID on HuggingFace (e.g., 'gpt2')
            api_key: HuggingFace API key. If None, will look for 'HF_API_KEY' in environment
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv('HF_API_KEY')
        self.api_url = api_url

        if not self.api_key or not self.api_url:
            raise ValueError("HuggingFace API key not found. Please provide it or set HF_API_KEY environment variable")
        
    def query(
        self,
        content: str,
    ) -> Dict[str, Any]:
        """
        Query the HuggingFace model with retry mechanism and proper error handling.
        
        Args:
            content: The input text to process
            
        Returns:
            Dict containing the model's response
            
        Raises:
            requests.exceptions.RequestException: If the request fails after all retries
        """
        tok = AutoTokenizer.from_pretrained("simplescaling/s1-32B")

        stop_token_ids = tok("<|im_end|>")["input_ids"]
        
        client = OpenAI(
            base_url = self.api_url,
            api_key = self.api_key
        )

        chat_completion = client.chat.completions.create(
            model="tgi",
            messages = [
                {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
                {"role": "user", "content": content}
            ],
            top_p=0.9,
            temperature=0.1,
            max_tokens=1024,
            stream=False,
            seed=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            logprobs=True
        )

        for message in chat_completion.choices:
            print(message.message.content, end = "")

# Example usage
if __name__ == "__main__":
    client = HFInferenceClient("https://p61e4xep6f1oyhp6.us-east-1.aws.endpoints.huggingface.cloud/v1")
    prompt = "How many r in raspberry"

    try:
        result = client.query(
            prompt,
        )
        print(result)
    except Exception as e:
        print(f"Error: {e}")