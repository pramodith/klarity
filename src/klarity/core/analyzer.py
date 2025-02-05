# analyzer.py
from typing import Any, List, Dict, Optional
from klarity.core.together_wrapper import TogetherModelWrapper
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
        insight_api_key: Optional[str] = None,
        insight_prompt_template: Optional[str] = None
    ):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.min_token_prob = min_token_prob
        
        # Initialize Together AI model if specified
        self.together_model = None
        self.insight_model = None
        self.insight_tokenizer = None
        
        if isinstance(insight_model, str) and insight_model.startswith("together:"):
            model_name = insight_model.replace("together:", "")
            self.together_model = TogetherModelWrapper(
                model_name=model_name,
                api_key=insight_api_key
            )
        else:
            self.insight_model = insight_model
            self.insight_tokenizer = insight_tokenizer
            
        self.insight_prompt_template = insight_prompt_template or """Analyze the uncertainty in this generation:

INPUT QUERY: {input_query}
GENERATED TEXT: {generated_text}

UNCERTAINTY METRICS:
{detailed_metrics}

Return a concise JSON analysis with uncertainty scores (0-100):
{{
    "scores": {{
        "overall_uncertainty": "<0-1>",
        "confidence_score": "<0-1>",
        "hallucination_risk": "<0-1>"
    }},
    "uncertainty_analysis": {{
        "high_uncertainty_parts": [
            {{
                "text": "<specific_text>",
                "why": "<brief_reason_for_uncertainty>"
            }}
        ],
        "main_issues": [
            {{
                "issue": "<specific_problem>",
                "evidence": "<brief_evidence>"
            }}
        ],
        "key_suggestions": [
            {{
                "what": "<specific_improvement>",
                "how": "<brief_implementation>"
            }}
        ]
    }}
}}

Be concise. Focus on specific content, not abstract metrics.

Analysis:"""

    def _calculate_raw_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate raw entropy of probability distribution"""
        max_entropy = np.log(len(probabilities))
        raw_entropy = entropy(probabilities)
        return raw_entropy / max_entropy if max_entropy != 0 else 0.0

    def _calculate_semantic_entropy(self, token_info: List[TokenInfo]) -> float:
        """Calculate semantic entropy based on token predictions"""
        if len(token_info) < 2:
            return 0.0
            
        tokens = [t.token for t in token_info]
        embeddings = self.embedding_model.encode(tokens)
        similarity_matrix = cosine_similarity(embeddings)
        semantic_groups = self._group_similar_tokens(similarity_matrix, token_info)
        group_probs = self._calculate_group_probabilities(semantic_groups, token_info)
        
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
            
        if total_prob > 0:
            for gid in group_probs:
                group_probs[gid] /= total_prob
                
        return group_probs

    def generate_overall_insight(
            self, 
            metrics_list: List[UncertaintyMetrics],
            input_query: Optional[str] = "",
            generated_text: Optional[str] = ""
        ) -> Optional[str]:
        """Generate overall insight for all collected metrics"""
        if not self.together_model and not (self.insight_model and self.insight_tokenizer):
            return None

        # Format metrics for the prompt
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

        all_metrics = "\n\n".join(detailed_metrics)
        prompt = self.insight_prompt_template.format(
            detailed_metrics=all_metrics,
            input_query=input_query,
            generated_text=generated_text
        )

        if self.together_model:
            return self.together_model.generate_insight(prompt)
            
        # Use original model
        inputs = self.insight_tokenizer(prompt, return_tensors="pt").to(self.insight_model.device)
        outputs = self.insight_model.generate(
            inputs.input_ids,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return self.insight_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Analysis:")[-1].strip()

    def analyze(self, request: UncertaintyAnalysisRequest) -> UncertaintyMetrics:
        """Analyze uncertainty for a single generation step"""
        probabilities = np.array([t.probability for t in request.token_info])
        raw_entropy = self._calculate_raw_entropy(probabilities)
        semantic_entropy = self._calculate_semantic_entropy(request.token_info)
        
        metrics = UncertaintyMetrics(
            raw_entropy=float(raw_entropy),
            semantic_entropy=float(semantic_entropy),
            token_predictions=request.token_info
        )

        return metrics