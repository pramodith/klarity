from typing import Dict, List, Optional, Callable, Tuple, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, LogitsProcessor, LogitsProcessorList
from .models import UncertaintyAnalysisRequest, TokenInfo, UncertaintyMetrics
from .core.analyzer import EntropyAnalyzer

class UncertaintyLogitsProcessor(LogitsProcessor):
    """Captures logits during generation for uncertainty analysis"""
    def __init__(self, estimator):
        self.captured_logits = []
        self.estimator = estimator
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.captured_logits.append(scores.detach().clone())
        return scores

class UncertaintyEstimator:
    def __init__(
        self, 
        top_k: int = 100,
        analyzer: Optional[EntropyAnalyzer] = None
    ):
        self.analyzer = analyzer if analyzer is not None else EntropyAnalyzer()
        self.top_k = top_k

    def get_logits_processor(self) -> LogitsProcessor:
        """Returns a logits processor that can be included in the user's generation pipeline"""
        return UncertaintyLogitsProcessor(self)

    def _process_logits(self, logits: torch.Tensor, tokenizer: PreTrainedTokenizer) -> List[TokenInfo]:
        """Process logits to get token information"""
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], self.top_k)
        top_logits = logits[0][top_indices]
        
        token_info = []
        for token_id, logit, prob in zip(top_indices, top_logits, top_probs):
            token = tokenizer.decode(token_id)
            token_info.append(TokenInfo(
                token=token,
                token_id=token_id.item(),
                logit=logit.item(),
                probability=prob.item()
            ))
        return token_info

    def analyze_generation(
        self,
        generation_output,
        tokenizer: PreTrainedTokenizer,
        processor: UncertaintyLogitsProcessor
    ) -> List[UncertaintyMetrics]:
        """Analyze uncertainty from a generation that used our logits processor"""
        generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        
        uncertainty_metrics = []
        for step, logits in enumerate(processor.captured_logits):
            token_info = self._process_logits(logits, tokenizer)
            
            request = UncertaintyAnalysisRequest(
                logits=logits[0].tolist(),
                prompt=generated_text[:step],
                model_id="user_model",  # User can pass this if needed
                token_info=token_info,
                metadata={
                    'step': step,
                    'total_steps': len(processor.captured_logits)
                }
            )
            
            metrics = self.analyzer.analyze(request)
            uncertainty_metrics.append(metrics)

        return uncertainty_metrics