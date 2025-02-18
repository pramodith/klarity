# core/together_wrapper.py
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from together import Together

from ..utils import TOGETHER_JSON_MODE_SUPPORTED_MODELS

class TogetherModelWrapper:
    """Wrapper for Together AI models supporting both text and vision"""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.is_vision_model = "Vision" in model_name or "vision" in model_name.lower()
        self.supports_json_mode = model_name in TOGETHER_JSON_MODE_SUPPORTED_MODELS

    def generate_insight_with_image(
        self,
        prompt: str,
        response_model: BaseModel,
        image_data: List[str],  # List of base64 encoded images
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Dict[str, Any]:
        """Generate insight using vision model"""
        if not self.is_vision_model:
            raise ValueError(f"Model {self.model_name} does not support vision inputs")

        # Create messages with text and images
        content = [{"type": "text", "text": prompt}]
        for img in image_data:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

        messages = [{"role": "user", "content": content}]

        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=messages, 
            temperature=temperature, 
            max_tokens=max_tokens, stream=False, 
            response_format={
                "type": "json_object",
                "schema": response_model.model_json_schema()
            } if self.supports_json_mode else None
        )

        try:
            # Try to parse as JSON first
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            # Return raw text if not valid JSON
            return {"text": response.choices[0].message.content}

    def generate_insight(self, prompt: str, response_model: BaseModel) -> str:
        """Generate text-only insight"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            response_format={
                "type": "json_object",
                "schema": response_model.model_json_schema()
            } if self.supports_json_mode else None
        )
        return response.choices[0].message.content
