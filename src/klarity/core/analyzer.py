#analyzer.py
from typing import List, Dict, Tuple
import numpy as np 
import torch
from scipy.stats import entropy
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ..models import UncertaintyMetrics, UncertaintyAnalysisRequest, TokenInfo
from transformers import PreTrainedTokenizer
from collections import defaultdict

class EntropyAnalyzer:
    def __init__(self, window_size: int = 5, min_token_prob: float = 0.01):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.window_size = window_size
        self.min_token_prob = min_token_prob

    def analyze(self, request: UncertaintyAnalysisRequest) -> UncertaintyMetrics:
        # Get sliding window of tokens for context
        window_tokens = self._get_sliding_window(request.token_info)
        
        # Calculate raw entropy as before
        probabilities = np.array([t.probability for t in request.token_info])
        raw_entropy = self._calculate_raw_entropy(probabilities)
        
        # Calculate semantic entropy based on token groups
        semantic_entropy = self._calculate_semantic_entropy(window_tokens, request.token_info)
        
        # Calculate confidence metrics
        coherence_score = self._calculate_coherence(window_tokens)
        divergence_score = self._calculate_prediction_divergence(request.token_info)
        
        # Estimate hallucination probability
        hallucination_prob = self._estimate_hallucination_probability(
            semantic_entropy,
            coherence_score,
            divergence_score,
            raw_entropy
        )
        
        return UncertaintyMetrics(
            raw_entropy=float(raw_entropy),
            semantic_entropy=float(semantic_entropy),
            coherence_score=float(coherence_score),
            divergence_score=float(divergence_score),
            hallucination_probability=float(hallucination_prob),
            token_predictions=request.token_info
        )

    def _get_sliding_window(self, token_info: List[TokenInfo]) -> List[str]:
        """Create meaningful token groups using sliding windows"""
        tokens = [t.token for t in token_info]
        windows = []
        
        for i in range(max(1, len(tokens) - self.window_size + 1)):
            window = tokens[i:i + self.window_size]
            windows.append(" ".join(window).strip())
            
        return windows
    
    def _calculate_raw_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate raw entropy of probability distribution"""
        max_entropy = np.log(len(probabilities))
        raw_entropy = entropy(probabilities)
        return raw_entropy / max_entropy if max_entropy != 0 else 0.0

    def _calculate_semantic_entropy(self, windows: List[str], token_info: List[TokenInfo]) -> float:
        """Calculate semantic entropy based on window embeddings and token probabilities"""
        if not windows:
            return 0.0
            
        # Get embeddings for all windows
        embeddings = self.embedding_model.encode(windows)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group similar meanings
        semantic_groups = self._group_similar_meanings(similarity_matrix, windows)
        
        # Calculate probability mass for each semantic group
        group_probs = self._calculate_group_probabilities(semantic_groups, token_info)
        
        # Calculate entropy over semantic groups
        if len(group_probs) > 1:
            return entropy(list(group_probs.values())) / np.log(len(group_probs))
        return 0.0

    def _group_similar_meanings(self, similarity_matrix: np.ndarray, windows: List[str], 
                              threshold: float = 0.8) -> Dict[int, List[int]]:
        """Group windows with similar semantic meanings"""
        groups = defaultdict(list)
        group_id = 0
        
        for i in range(len(windows)):
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
            group_prob = sum(token_info[i].probability for i in indices 
                           if i < len(token_info))
            group_probs[gid] = group_prob
            total_prob += group_prob
            
        # Normalize probabilities
        if total_prob > 0:
            for gid in group_probs:
                group_probs[gid] /= total_prob
                
        return group_probs

    def _calculate_coherence(self, windows: List[str]) -> float:
        """Calculate semantic coherence of the generated text"""
        if len(windows) < 2:
            return 1.0
            
        embeddings = self.embedding_model.encode(windows)
        similarities = cosine_similarity(embeddings)
        
        # Calculate average similarity between consecutive windows
        consecutive_similarities = np.diagonal(similarities, offset=1)
        return float(np.mean(consecutive_similarities))

    def _calculate_prediction_divergence(self, token_info: List[TokenInfo]) -> float:
        """Calculate how much top predictions diverge from each other"""
        top_k = 5  # Consider top 5 predictions
        
        if len(token_info) < top_k:
            return 0.0
            
        top_probs = [t.probability for t in token_info[:top_k]]
        max_divergence = np.log(top_k)
        current_divergence = entropy(top_probs)
        
        return current_divergence / max_divergence if max_divergence > 0 else 0.0

    def _estimate_hallucination_probability(self, semantic_entropy: float, 
                                          coherence: float, divergence: float, 
                                          raw_entropy: float) -> float:
        """Estimate probability of hallucination based on multiple metrics"""
        # Weighted combination of factors that might indicate hallucination
        weights = {
            'semantic_entropy': 0.3,
            'coherence': 0.3,
            'divergence': 0.2,
            'raw_entropy': 0.2
        }
        
        hallucination_score = (
            weights['semantic_entropy'] * semantic_entropy +
            weights['coherence'] * (1 - coherence) +  # Lower coherence -> higher probability
            weights['divergence'] * divergence +
            weights['raw_entropy'] * raw_entropy
        )
        
        # Sigmoid to get probability between 0 and 1
        return 1 / (1 + np.exp(-5 * (hallucination_score - 0.5)))