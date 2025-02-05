# estimator.py
from typing import Dict, List, Optional
import numpy as np
import torch
from transformers import PreTrainedTokenizer, LogitsProcessor
from .models import UncertaintyAnalysisRequest, TokenInfo, UncertaintyMetrics, UncertaintyAnalysisResult
from .core.analyzer import EntropyAnalyzer

class UncertaintyLogitsProcessor(LogitsProcessor):
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
        return UncertaintyLogitsProcessor(self)

    def _process_logits(self, logits: torch.Tensor, tokenizer: PreTrainedTokenizer) -> List[TokenInfo]:
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
    ) -> UncertaintyAnalysisResult:
        generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        
        # Collect all token metrics first
        all_metrics = []
        for step, logits in enumerate(processor.captured_logits):
            token_info = self._process_logits(logits, tokenizer)
            
            request = UncertaintyAnalysisRequest(
                logits=logits[0].tolist(),
                prompt=generated_text[:step],
                model_id="user_model",
                token_info=token_info,
                metadata={'step': step}
            )
            
            # Create metrics without insight
            metrics = UncertaintyMetrics(
                raw_entropy=self.analyzer._calculate_raw_entropy(
                    np.array([t.probability for t in token_info])
                ),
                semantic_entropy=self.analyzer._calculate_semantic_entropy(token_info),
                token_predictions=token_info
            )
            all_metrics.append(metrics)

        # Generate single insight using all collected data
        overall_insight = self.analyzer.generate_overall_insight(
            all_metrics,
            input_query=generated_text[:step],  # or however you want to get the input query
            generated_text=generated_text  # the full generated text
        )
        
        return UncertaintyAnalysisResult(
            token_metrics=all_metrics,
            overall_insight=overall_insight
        )