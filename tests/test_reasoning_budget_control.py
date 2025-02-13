import unittest
from unittest.mock import Mock, patch
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.klarity.benchmarks.reasoning_budget_control import VLLMClient


class MockOutput:
    def __init__(self, text: str, token_ids: List[int], num_generated_tokens: int):
        self.text = text
        self.token_ids = token_ids
        self.num_generated_tokens = num_generated_tokens


class MockOutputGeneration:
    def __init__(self, text: str, token_ids: List[int], num_generated_tokens: int):
        self.outputs = [MockOutput(text, token_ids, num_generated_tokens)]


class TestVLLMClient(unittest.TestCase):
    def setUp(self):
        # Mock the LLM and tokenizer
        self.mock_llm = Mock(spec=LLM)
        self.mock_tokenizer = Mock(spec=AutoTokenizer)

        # Create patches
        self.llm_patcher = patch("src.klarity.benchmarks.reasoning_budget_control.LLM", return_value=self.mock_llm)
        self.tokenizer_patcher = patch("src.klarity.benchmarks.reasoning_budget_control.AutoTokenizer")

        # Start patches
        self.llm_mock = self.llm_patcher.start()
        self.tokenizer_mock = self.tokenizer_patcher.start()
        self.tokenizer_mock.from_pretrained.return_value = self.mock_tokenizer

        # Initialize client with mocked dependencies
        self.client = VLLMClient()

    def tearDown(self):
        # Stop patches
        self.llm_patcher.stop()
        self.tokenizer_patcher.stop()

    def test_extract_answer_valid(self):
        """Test extract_answer with valid input"""
        test_cases = [
            ("The answer is boxed{42}", "42"),
            ("Multiple boxed{1} answers boxed{2}", "2"),
            ("No boxes", ""),
            ("Incomplete boxed{", ""),
            ("boxed{nested boxed{42}}", "42"),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.client.extract_answer(input_text)
                self.assertEqual(result, expected)

    def test_query_with_extra_wait(self):
        """Test query_with_extra_wait with mocked model and tokenizer"""
        # Setup mock responses
        mock_text = "Test output boxed{42}"
        mock_tokens = [1, 2, 3, 4]
        mock_output = [MockOutputGeneration(mock_text, mock_tokens)]
        self.mock_llm.generate.return_value = mock_output

        # Mock tokenizer behavior
        self.mock_tokenizer.decode.return_value = mock_text

        # Test with different numbers of waits
        test_cases = [(0, 1), (1, 2), (2, 3)]  # (num_waits, expected_generations)

        for num_waits, expected_generations in test_cases:
            with self.subTest(num_waits=num_waits):
                prompt = "Test prompt"
                sampling_params = SamplingParams()

                generated_texts, total_tokens, answer = self.client.query_with_extra_wait(
                    prompt, num_waits, sampling_params
                )

                # Verify the number of generations matches expected
                self.assertEqual(len(generated_texts), expected_generations)
                self.assertIsInstance(total_tokens, int)
                self.assertEqual(answer, "42}")  # Should extract from boxed{42}

    def test_compute_math_accuracy(self):
        """Test compute_math_accuracy with various test cases"""
        test_cases = [
            (["42", "7"], ["42", "7"], 1.0),  # All correct
            (["42", "8"], ["42", "7"], 0.5),  # Half correct
            ([], [], 0.0),  # Empty lists
        ]

        for predictions, ground_truths, expected_accuracy in test_cases:
            with self.subTest(predictions=predictions, ground_truths=ground_truths):
                accuracy = self.client.compute_math_accuracy(predictions, ground_truths)
                self.assertEqual(accuracy, expected_accuracy)

    def test_compute_math_accuracy_different_lengths(self):
        """Test that compute_math_accuracy raises ValueError when lists have different lengths"""
        predictions = ["42", "7"]
        ground_truths = ["42", "7", "8"]

        with self.assertRaises(ValueError) as _:
            self.client.compute_math_accuracy(predictions, ground_truths)

    def test_add_system_prompt(self):
        """Test add_system_prompt functionality"""
        test_prompt = "What is 2+2?"
        result = self.client.add_system_prompt(test_prompt)
        desired_result = (
            "Please reason step by step, and put your final answer within \\boxed\{\}."
            "Solve the following problem:" + "What is 2+2?"
        )
        self.assertEqual(result, desired_result)


if __name__ == "__main__":
    unittest.main()
