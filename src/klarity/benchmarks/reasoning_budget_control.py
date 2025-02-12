import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from vllm import LLM, SamplingParams
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

class VLLMClient:

    def __init__(self, model:str = "agentica-org/DeepScaleR-1.5B-Preview", tensor_parallel_size: int = 1):
        self.model = LLM(
            model,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            dtype="bfloat16",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def add_system_prompt(self, prompt: str):
        prompt = "How many r in raspberry"
        prompt = f"{self.tokenizer.bos_token}You are a helpful assistant. Solve the following problem: " + prompt + 
        return prompt

    def query(self, query: List[str], min_tokens: int = 0, max_tokens: int = 32768):
        """
        Query the model with the given prompt.

        Args:
        - prompt (str): The prompt to generate text for.
        - min_tokens (int, optional): The minimum number of tokens to generate. Defaults to 0.
        - max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 32768.

        Returns:
        - str: The generated text.
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            stop_token_ids=self.tokenizer("</think>")["input_ids"]
        )

        o = self.model.generate(prompt, sampling_params=sampling_params)
        num_generated_tokens = [len(o[i].outputs[0].token_ids) for i in range(len(o))]
        print(f"Average number of tokens: {sum(num_generated_tokens) / len(num_generated_tokens)}")
        print(o[0].outputs[0].text)
        return num_generated_tokens

# Example usage
if __name__ == "__main__":
    # client = HFInferenceClient("https://p61e4xep6f1oyhp6.us-east-1.aws.endpoints.huggingface.cloud/v1")
    # prompt = "How many r in raspberry"

    # try:
    #     result = client.query(
    #         prompt,
    #     )
    #     print(result)
    # except Exception as e:
    #     print(f"Error: {e}")

    vllm_client = VLLMClient()
    query = "How many r in raspberry"
    prompt = vllm_client.add_system_prompt(query)
    result = vllm_client.query([prompt])