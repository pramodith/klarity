# analyzer.py
from typing import Any, List, Dict, Optional
import numpy as np
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from ..models import TokenInfo, UncertaintyAnalysisRequest, UncertaintyMetrics

class EntropyAnalyzer:
    def __init__(
        self, 
        min_token_prob: float = 0.01,
        insight_model: Optional[Any] = None,
        insight_tokenizer: Optional[Any] = None,
        insight_prompt_template: Optional[str] = None
    ):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.min_token_prob = min_token_prob
        self.insight_model = insight_model
        self.insight_tokenizer = insight_tokenizer
        self.insight_prompt_template = insight_prompt_template or """Analyze step-by-step metrics:

{detailed_metrics}

Provide analysis:

1. HIGH/LOW POINTS
- Steps with highest entropy (>0.6)
- Steps with lowest entropy (<0.2)

2. KEY DECISIONS
- Most uncertain choices (steps & options)
- Most certain predictions (>0.9)

3. ENTROPY COMPARISON
- Raw vs semantic patterns
- Notable divergences

4. RELIABILITY
- Major uncertainty areas
- Potential issues

Reference specific steps and values.

Analysis:"""

    def analyze(self, request: UncertaintyAnalysisRequest) -> UncertaintyMetrics:
        # Calculate raw entropy
        probabilities = np.array([t.probability for t in request.token_info])
        raw_entropy = self._calculate_raw_entropy(probabilities)
        
        # Calculate semantic entropy based on token predictions
        semantic_entropy = self._calculate_semantic_entropy(request.token_info)
        
        metrics = UncertaintyMetrics(
            raw_entropy=float(raw_entropy),
            semantic_entropy=float(semantic_entropy),
            token_predictions=request.token_info
        )

        if self.insight_model and self.insight_tokenizer:
            metrics.insight = self.generate_overall_insight(metrics)
        
        return metrics

    def _calculate_raw_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate raw entropy of probability distribution"""
        max_entropy = np.log(len(probabilities))
        raw_entropy = entropy(probabilities)
        return raw_entropy / max_entropy if max_entropy != 0 else 0.0

    def _calculate_semantic_entropy(self, token_info: List[TokenInfo]) -> float:
        """Calculate semantic entropy based on token predictions"""
        if len(token_info) < 2:
            return 0.0
            
        # Get embeddings for all predicted tokens
        tokens = [t.token for t in token_info]
        embeddings = self.embedding_model.encode(tokens)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group similar tokens
        semantic_groups = self._group_similar_tokens(similarity_matrix, token_info)
        
        # Calculate probability mass for each semantic group
        group_probs = self._calculate_group_probabilities(semantic_groups, token_info)
        
        # Calculate entropy over semantic groups
        if len(group_probs) > 1:
            return entropy(list(group_probs.values())) / np.log(len(group_probs))
        return 0.0

    def _group_similar_tokens(self, similarity_matrix: np.ndarray, 
                            token_info: List[TokenInfo],
                            threshold: float = 0.8) -> Dict[int, List[int]]:
        """Group tokens with similar semantic meanings"""
        groups = defaultdict(list)
        group_id = 0
        
        for i in range(len(token_info)):
            assigned = False
            for gid in groups:
                if any(similarity_matrix[i][j] > threshold for j in groups[gid]):
                    groups[gid].append(i)
                    assigned = True
                    break
            if not assigned:
                groups[group_id] = [i]
                group_id += 1
                
        return groups

    def _calculate_group_probabilities(self, semantic_groups: Dict[int, List[int]], 
                                     token_info: List[TokenInfo]) -> Dict[int, float]:
        """Calculate probability mass for each semantic group"""
        group_probs = defaultdict(float)
        total_prob = 0.0
        
        for gid, indices in semantic_groups.items():
            group_prob = sum(token_info[i].probability for i in indices)
            group_probs[gid] = group_prob
            total_prob += group_prob
            
        # Normalize probabilities
        if total_prob > 0:
            for gid in group_probs:
                group_probs[gid] /= total_prob
                
        return group_probs
    
    def generate_overall_insight(self, metrics_list: List[UncertaintyMetrics]) -> Optional[str]:
        if not self.insight_model or not self.insight_tokenizer:
            return None

        # Build detailed metrics string
        detailed_metrics = []
        for idx, metrics in enumerate(metrics_list):
            top_predictions = [
                f"{t.token} ({t.probability:.3f})"
                for t in metrics.token_predictions[:3]
            ]
            
            step_metrics = (
                f"Step {idx}:\n"
                f"- Raw Entropy: {metrics.raw_entropy:.4f}\n"
                f"- Semantic Entropy: {metrics.semantic_entropy:.4f}\n"
                f"- Top 3 Predictions: {' | '.join(top_predictions)}"
            )
            detailed_metrics.append(step_metrics)

        # Combine all metrics with proper spacing
        all_metrics = "\n\n".join(detailed_metrics)

        # Generate prompt with detailed metrics
        prompt = self.insight_prompt_template.format(detailed_metrics=all_metrics)

        # Generate insight
        inputs = self.insight_tokenizer(prompt, return_tensors="pt").to(self.insight_model.device)
        outputs = self.insight_model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return self.insight_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Analysis:")[-1].strip()