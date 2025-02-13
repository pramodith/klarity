from typing import Dict, List, Any, Optional
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import argparse
import os

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

    WAIT_STR: str = "Wait, "
    def __init__(
        self, 
        model:str = "agentica-org/DeepScaleR-1.5B-Preview", 
        tensor_parallel_size: int = 1, 
        ):
        
        self.model = LLM(
            model,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            dtype="bfloat16",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def add_system_prompt(self, prompt: str):
        prompt = f"You are a helpful assistant."\
            "Please reason step by step, and put your final answer within \\boxed\{\}."\
            "Solve the following problem:"\
            + prompt + " <think>\n"
        return prompt
    
    def extract_answer(self, text: str):
        # Extract the answer between \boxed{ and }
        return text[text.find("\\boxed\\{"):text.find("}")+1]

    def query(
        self, 
        query: List[str], 
        min_tokens: int = 0, 
        max_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ):
        """
        Query the model with the given prompt.

        Args:
        - prompt (str): The prompt to generate text for.
        - min_tokens (int, optional): The minimum number of tokens to generate. Defaults to 0.
        - max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 32768.
        - temperature (float, optional): The temperature to use. Defaults to 0.6.
        - top_p (float, optional): The top-p to use. Defaults to 0.95.

        Returns:
        - str: The generated text.
        """
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temperature,
            top_p=top_p,
            # stop_token_ids=self.tokenizer("</think>")["input_ids"]
        )

        o = self.model.generate(query, sampling_params=sampling_params)
        num_generated_tokens = [len(o[i].outputs[0].token_ids) for i in range(len(o))]
        print(f"Average number of tokens: {sum(num_generated_tokens) / len(num_generated_tokens)}")
        final_answers = [self.extract_answer(o[i].outputs[0].text) for i in range(len(o))]
        print(o[0].outputs[0].text)
        return num_generated_tokens, final_answers
    
    def get_vllm_output(self, output_generation):
        generated_text = output_generation.outputs[0].text
        num_generated_tokens = len(output_generation.outputs[0].token_ids)
        return generated_text, num_generated_tokens

    def query_with_extra_wait(
        self, 
        prompt: str, 
        num_waits: int = 1,
        sampling_params: SamplingParams = SamplingParams(), 
        temperature: float = 0.6,
        top_p: float = 0.95
    ):
        """
        Query the model with the given prompt.

        Args:
        - prompt (str): The prompt to generate text for.
        - min_tokens (int, optional): The minimum number of tokens to generate. Defaults to 0.
        - max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 32768.
        - temperature (float, optional): The temperature to use. Defaults to 0.6.
        - top_p (float, optional): The top-p to use. Defaults to 0.95.

        Returns:
        - str: The generated text and the number of tokens generated.
        """

        if num_waits < 0:
            raise ValueError("num_waits must be non-negative")
        elif num_waits == 0:
            stop_token_ids = None
        else:
            stop_token_ids = self.tokenizer("</think>")["input_ids"]

        generated_texts = []
        num_generated_tokens = []

        for wait_ind in range(num_waits+1):
            if wait_ind == num_waits - 1:
                sampling_params.stop_token_ids = stop_token_ids

            o = self.model.generate(prompt, sampling_params=sampling_params)
            gt, nt = self.get_vllm_output(o[0])
            # Force reflection by adding another wait
            prompt += gt + self.WAIT_STR
            generated_texts.append(gt)
            num_generated_tokens.append(nt)
            

        with open(f"generated_texts_{num_waits}.txt", "w") as f:
            f.write("||".join(generated_texts))

        # Get the entire generated text and the total generated token count
        return " ".join(generated_texts), sum(num_generated_tokens)
    
    def load_aime_dataset(self, dataset_path: str = "HuggingFaceH4/aime_2024") -> List[str]:
        dataset = load_dataset(dataset_path)
        dataset = dataset["train"]
        final_dataset = []
        for row in dataset:
            final_dataset.append({"query": row["problem"], "answer": row["answer"]})

        return final_dataset
    
    def compute_math_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Compute the accuracy of the predictions compared to the ground truth for math problems.

        Args:
        - predictions (List[str]): The predicted answers.
        - ground_truths (List[str]): The ground truth answers.

        Returns:
        - float: The accuracy of the predictions.
        """
        correct = 0
        total = len(predictions)
        for pred, gt in zip(predictions, ground_truths):
            if float(pred) == float(gt):
                correct += 1
        return correct / total
    
    def main(
        num_waits: int = 0,
        max_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ):  
        """
        Main function to run the benchmark.

        Args:
        - num_waits (int, optional): The number of extra waits to add. Defaults to 0.
        - max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 32768.
        - temperature (float, optional): The temperature to use. Defaults to 0.6.
        - top_p (float, optional): The top-p to use. Defaults to 0.95.

        Returns:
        - None
        """
        dataset = vllm_client.load_aime_dataset()
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        predicted_answers = []
        gt_answers = []
        for row in dataset:
            query = row["query"]
            answer = row["answer"]
            prompt = vllm_client.add_system_prompt(query)
            generated_response, total_generated_tokens = vllm_client.query_with_extra_wait(prompt, num_waits)
            print(generated_response)
            predicted_answers.append(generated_response)
            gt_answers.append(answer)

        math_accuracy = self.compute_math_accuracy(predicted_answers, gt_answers)
        print(f"Math Accuracy: {math_accuracy * 100:.2f}%")
        return math_accuracy

# Example usage
if __name__ == "__main__":
    # client = HFInferenceClient("https://p61e4xep6f1oyhp6.us-east-1.aws.endpoints.huggingface.cloud/v1")
    # prompt = "How many r in raspberry"
    # client = HFInferenceClient("https://p61e4xep6f1oyhp6.us-east-1.aws.endpoints.huggingface.cloud/v1")
    # prompt = "How many r in raspberry"

    # try:
    #     result = client.query(
    #         prompt,
    #     )
    #     print(result)
    # except Exception as e:
    #     print(f"Error: {e}")

    # Accept args from the user 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_waits", type=int, default=1, help="Number of extra waits to add")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for the query")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for the query")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for the query")
    
    args = parser.parse_args()
    vllm_client = VLLMClient()

    vllm_client.main(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )