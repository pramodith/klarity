import unittest
from klarity.benchmarks.reasoning_budget_control import VLLMClient

class TestVLLMClient(unittest.TestCase):
    def setUp(self):
        self.client = VLLMClient()

    def test_extract_answer(self):
        # Test case 1: Normal case with boxed content
        input_text = "Some text \\boxed{42} more text"
        expected = "\\boxed{42}"
        self.assertEqual(self.client.extract_answer(input_text), expected)

        # Test case 2: Multiple boxed contents (should extract until first closing brace)
        input_text = "\\boxed{10} and \\boxed{20}"
        expected = "\\boxed{10}"
        self.assertEqual(self.client.extract_answer(input_text), expected)

        # Test case 3: No boxed content
        input_text = "Just some regular text"
        expected = ""
        self.assertEqual(self.client.extract_answer(input_text), expected)

        # Test case 4: Empty string
        input_text = ""
        expected = ""
        self.assertEqual(self.client.extract_answer(input_text), expected)

if __name__ == '__main__':
    unittest.main()
