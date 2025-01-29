# estimator.py
from typing import Dict, List, Optional, Callable
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .models import UncertaintyAnalysisRequest
from .core.analyzer import EntropyAnalyzer

# estimator.py
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
            logits = outputs.logits[:, -1, :]  # Keep as tensor
            
            # Get top k tokens and their probabilities
            probs = torch.softmax(logits, dim=-1)
            top_k = 20
            top_probs, top_indices = torch.topk(probs[0], top_k)
            
            # Get the actual tokens for these predictions
            predicted_tokens = [tokenizer.decode(idx.item()) for idx in top_indices]

        request = UncertaintyAnalysisRequest(
            logits=logits.flatten().tolist(),
            prompt=prompt,
            model_id=model.config._name_or_path,
            metadata={
                'top_tokens': predicted_tokens,
                'top_probs': top_probs.tolist()
            }
        )
        
        return self.analyzer.analyze(request)