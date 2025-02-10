# together_wrapper.py
from typing import Optional
from together import Together


class TogetherModelWrapper:
    """Wrapper for Together AI models to provide consistent interface"""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.client = Together(api_key=api_key)
        self.model_name = model_name

    def generate_insight(self, prompt: str) -> str:
        """Generate insight using Together AI model"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=800,
        )
        return response.choices[0].message.content
