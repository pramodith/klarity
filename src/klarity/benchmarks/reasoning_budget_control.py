from datasets import load_dataset
from enum import Enum
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from loguru import logger

import argparse
import numpy as np
import sys

logger.add("logs/reasoning_budget_control.log")

class DatasetType(str, Enum):
    AIME = "aime"
    MATH = "math"

class VLLMClient:

    WAIT_STR: str = " Wait, "
    THINK_START_STR = "<think>"
    THINK_END_STR = "</think>"
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

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model)
    
    def add_system_prompt(self, prompt: str):
        """
        Add system prompt to the prompt.

        Args:
            prompt (str): The prompt to add the system prompt to.

        Returns:
            str: The prompt with the system prompt added.
        """
        prompt = "Please reason step by step, and put your final answer within \\boxed\{\}."\
            "Solve the following problem:"\
            + prompt
        return prompt

    def tokenize_prompt(self, prompt: str) -> str:
        """
        Tokenize the prompt using the tokenizer.

        Args:
            prompt (str): The prompt to tokenize.

        Returns:
            str: The text of the prompt after applying the chat template.
        """
        prompt = self.add_system_prompt(prompt)
        tokens_prompt = TokensPrompt(
            prompt_token_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        ))

        return self.tokenizer.decode(tokens_prompt["prompt_token_ids"], skip_special_tokens=False) + self.THINK_START_STR

    def extract_answer(self, text: str):
        """
        Extract the answer between boxed{ and }.
        
        Args:
            text (str): The text to extract the answer from.
        
        Returns:
            str: The answer extracted from the text.
        """
        try:
            last_occurrence = text.rindex("boxed{")
            end_pos = text.find("}", last_occurrence)
            if end_pos != -1:
                return text[last_occurrence+len("boxed{"):end_pos]
            return ""
        except ValueError:
            return ""
    
    def get_vllm_output(self, output_generation):
        """
        Get the output of the model.

        Args:
            output_generation (OutputGeneration): The output generation object.

        Returns:
            Tuple[str, List[int], int]: The generated text, the generated tokens, and the number of generated tokens.
        """
        generated_text = output_generation.outputs[0].text
        generated_tokens = output_generation.outputs[0].token_ids
        num_generated_tokens = len(output_generation.outputs[0].token_ids)
        return generated_text, generated_tokens, num_generated_tokens

    def query_with_extra_wait(
        self, 
        prompt: str, 
        num_waits: int = 1,
        sampling_params: SamplingParams = SamplingParams(), 
    ):
        """
        Query the model with the given prompt.

        Args:
        - prompt (str): The prompt to generate text for.
        - num_waits (int, optional): The number of extra waits to add. Defaults to 1.
        - sampling_params (SamplingParams, optional): The sampling parameters to use. Defaults to SamplingParams().

        Returns:
        - str: The generated text and the number of tokens generated.
        """

        if num_waits < 0:
            raise ValueError("num_waits must be non-negative")
        elif num_waits == 0:
            stop = None
        else:
            stop = [self.THINK_END_STR]

        sampling_params.stop = stop

        # Tokenize prompt
        generated_texts = []
        num_generated_tokens = []

        for wait_ind in range(num_waits+1):
            if wait_ind == num_waits:
                # If this is the last wait, set stop to None to ensure full generation
                sampling_params.stop = None
            try:
                # Generate text using the model
                outputs = self.model.generate(prompt, sampling_params=sampling_params)
                
                # Process model outputs
                generated_texts_batch, token_ids, token_counts = zip(*[
                    self.get_vllm_output(output) for output in outputs
                ])
                
                # Decode token IDs to text
                decoded_texts = [
                    self.tokenizer.decode(tokens, skip_special_tokens=False)
                    for tokens in token_ids
                ]
                
                # Add wait string if needed
                if wait_ind < num_waits:
                    decoded_texts = [text + self.WAIT_STR for text in decoded_texts]
                
                # Store results
                generated_texts.append(list(generated_texts_batch))
                num_generated_tokens.append(list(token_counts))
                
                # Prepare next prompt
                prompt = self.tokenizer(decoded_texts)
                
            except Exception as e:
                # Log the error and handle gracefully
                logger.error(f"Error during text generation: {str(e)}")
                batch_size = len(prompt)
                generated_texts.append([""] * batch_size)
                num_generated_tokens.append([-1] * batch_size)

        # We want to find the total number of tokens generated per input prompt so we need to sum
        # along the wait axis.
        num_generated_tokens = np.array(num_generated_tokens).sum(0)
        predicted_answers = [self.extract_answer(text) for text in decoded_texts]

        # with open(f"generated_texts_{num_waits}.txt", "w") as f:
        #     f.write("||".join(generated_texts))

        # Get the entire generated text and the total generated token count
        return decoded_texts, sum(num_generated_tokens), predicted_answers
    
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
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        if len(predictions) == 0:
            return 0.0
        
        correct = 0
        total = len(predictions)
        for pred, gt in zip(predictions, ground_truths):
            try:
                equal = float(pred) == float(gt)
                if equal:
                    correct += 1
            except ValueError:
                pass
        return correct / total
    
    def main(
        self,
        num_waits: int = 0,
        num_sample_responses: int = 1,
        dataset_type: DatasetType = DatasetType.AIME,
        max_tokens: int = 16384,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ):  
        """
        Main function to run the benchmark.

        Args:
        - dataset_type (DatasetType, optional): The type of dataset to use. Defaults to DatasetType.AIME.
        - num_waits (int, optional): The number of extra waits to add. Defaults to 0.
        - max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 16384.
        - temperature (float, optional): The temperature to use. Defaults to 0.6.
        - top_p (float, optional): The top-p to use. Defaults to 0.95.

        Returns:
        - None
        """
        if dataset_type == DatasetType.AIME.value:
            dataset = vllm_client.load_aime_dataset()
        # elif dataset_type == DatasetType.MATH.value:
        #     dataset = vllm_client.load_math_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        accuracy_across_samples = []
        num_generated_tokens = 0
        for i in tqdm(range(num_sample_responses), desc="Sample number", total=num_sample_responses):
            predicted_answers = []
            gt_answers = []
            queries = []
            for row in tqdm(dataset, desc="Dataset row", total=len(dataset)):
                queries.append(row["query"])
                gt_answers.append(row["answer"])

            prompts = [self.add_system_prompt(query) for query in queries]
            inputs = [self.tokenize_prompt(prompt) for prompt in prompts]
            
            generated_responses, total_generated_tokens, predicted_answers = self.query_with_extra_wait(inputs, num_waits, sampling_params)
            
            if total_generated_tokens == -1:
                continue
            num_generated_tokens += total_generated_tokens

            accuracy = self.compute_math_accuracy(predicted_answers, gt_answers)
            accuracy_across_samples.append(accuracy)
            logger.debug(f"Accuracy for sample {i}: {accuracy * 100}%")
            logger.debug(f"Average number of tokens generated: {num_generated_tokens / len(dataset)}")

        pass_1 = sum(accuracy_across_samples) / len(accuracy_across_samples) * 100
        logger.debug(f"Pass@1 for {num_sample_responses}: {pass_1}%")
        logger.debug(f"predicted_answers: {predicted_answers}")

if __name__ == "__main__":
    # Accept args from the user 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_waits", type=int, default=0, help="Number of extra waits to add")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens for the query")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for the query")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for the query")
    parser.add_argument(
        "--dataset_type", 
        type=DatasetType, 
        default=DatasetType.AIME.value, 
        help="Dataset type has to be either aime or math",
        choices=[DatasetType.AIME.value, DatasetType.MATH.value]
    )
    args = parser.parse_args()
    if args.dataset_type == DatasetType.AIME.value:
        dataset_type = DatasetType.AIME
    elif args.dataset_type == DatasetType.MATH.value:
        dataset_type = DatasetType.MATH

    args = parser.parse_args()
    vllm_client = VLLMClient()

    vllm_client.main(
        num_waits=args.num_waits,
        dataset_type=dataset_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )