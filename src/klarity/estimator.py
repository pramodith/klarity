from typing import Dict, List, Optional, Callable
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .models import UncertaintyAnalysisRequest, ClosedSourceAnalysisRequest
from .core.analyzer import EntropyAnalyzer

class UncertaintyEstimator:
    def __init__(self):
        self.analyzer = EntropyAnalyzer()

    def estimate(
        self,
        prompt: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ) -> Dict:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :].flatten().tolist()

        request = UncertaintyAnalysisRequest(
            logits=logits,
            prompt=prompt,
            model_id=model.config._name_or_path
        )
        
        return self.analyzer.analyze(request)

    def estimate_closed_source(
        self,
        prompt: str,
        model_id: str,
        prompts_generator: Optional[Callable[[str], List[str]]] = None,
        n_variations: int = 5,
        api_call_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict:
        if not api_call_fn:
            raise ValueError("api_call_fn is required for closed source models")
        
        # Generate prompt variations
        if prompts_generator:
            prompt_variations = prompts_generator(prompt)
        else:
            prompt_variations = [
                prompt,
                f"Could you help me with this: {prompt}",
                f"I'd like to know: {prompt}",
                f"Please tell me about: {prompt}",
                f"I have a question: {prompt}"
            ][:n_variations]
        
        # Get responses for each prompt variation
        responses = []
        for p in prompt_variations:
            try:
                response = api_call_fn(p)
                responses.append(response)
            except Exception as e:
                print(f"Error getting response for prompt: {e}")
                continue
                
        request = ClosedSourceAnalysisRequest(
            prompt=prompt,
            model_id=model_id,
            prompt_variations=prompt_variations,
            responses=responses
        )
        
        return self.analyzer.analyze_closed_source(request)