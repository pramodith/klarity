from typing import Dict
import torch
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from ..models import UncertaintyMetrics, UncertaintyAnalysisRequest

class EntropyAnalyzer:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def analyze(self, request: UncertaintyAnalysisRequest) -> Dict:
        # Get the top tokens and their probabilities from metadata
        top_tokens = request.metadata['top_tokens']
        top_probs = request.metadata['top_probs']
        
        # Normalize probabilities
        probabilities = np.array(top_probs)
        probabilities = probabilities / np.sum(probabilities)
        
        # Calculate raw entropy
        raw_entropy = entropy(probabilities)
        max_entropy = np.log(len(probabilities))
        normalized_raw_entropy = raw_entropy / max_entropy if max_entropy != 0 else 0.0
        
        # Get embeddings of actual predicted tokens
        embeddings = self.embedding_model.encode(top_tokens)
        
        # Calculate similarity matrix
        similarity_matrix = np.array([[np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) 
                                    for b in embeddings] for a in embeddings])
        
        # Cluster based on semantic similarity
        clusters, cluster_quality = self._get_semantic_clusters(embeddings)
        
        # Calculate semantic entropy
        cluster_probs = {}
        for cluster_id, prob in zip(clusters, probabilities):
            cluster_probs[cluster_id] = cluster_probs.get(cluster_id, 0) + prob
            
        if len(cluster_probs) > 1:
            cluster_probs_array = np.array(list(cluster_probs.values()))
            semantic_entropy = entropy(cluster_probs_array) / np.log(len(cluster_probs))
        else:
            semantic_entropy = 0.0
            
        return UncertaintyMetrics(
            raw_entropy=float(normalized_raw_entropy),
            semantic_entropy=float(semantic_entropy),
            cluster_quality=float(cluster_quality),
            n_clusters=int(len(cluster_probs))
        )

    def _get_semantic_clusters(self, embeddings: np.ndarray) -> tuple:
        if len(embeddings) < 3:
            return np.zeros(len(embeddings)), 0.0
        
        # Calculate affinity matrix
        affinity_matrix = 1 - np.array([[1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) 
                                    for b in embeddings] for a in embeddings])
        
        # Use spectral clustering
        clustering = SpectralClustering(
            n_clusters=min(5, len(embeddings)-1),
            affinity='precomputed',
            random_state=42
        )
        
        try:
            clusters = clustering.fit_predict(affinity_matrix)
            if len(np.unique(clusters)) > 1:
                quality = silhouette_score(embeddings, clusters, metric='cosine')
            else:
                quality = 0.0
        except:
            return np.zeros(len(embeddings)), 0.0
            
        return clusters, quality