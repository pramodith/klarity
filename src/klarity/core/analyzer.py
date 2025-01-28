import torch
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from ..models import UncertaintyMetrics, UncertaintyAnalysisRequest, ClosedSourceAnalysisRequest

class EntropyAnalyzer:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def analyze(self, request: UncertaintyAnalysisRequest) -> Dict:
        logits_tensor = torch.tensor(request.logits).unsqueeze(0)
        probs = torch.softmax(logits_tensor, dim=-1)
        
        top_k = 20
        top_k_probs, _ = torch.topk(probs[0], top_k)
        probabilities = top_k_probs.numpy()
        
        raw_entropy = entropy(probabilities)
        embeddings = self.embedding_model.encode([str(i) for i in range(len(probabilities))])
        clusters, cluster_quality = self._get_semantic_clusters(embeddings)
        
        cluster_probs = {}
        for cluster_id, prob in zip(clusters, probabilities):
            cluster_probs[cluster_id] = cluster_probs.get(cluster_id, 0) + prob
        
        semantic_entropy = entropy(list(cluster_probs.values()))
        mean_info = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        varentropy = np.sum(probabilities * ((-np.log2(probabilities + 1e-10) - mean_info) ** 2))
        
        return UncertaintyMetrics(
            raw_entropy=float(raw_entropy),
            semantic_entropy=float(semantic_entropy),
            varentropy=float(varentropy),
            cluster_quality=float(cluster_quality),
            n_clusters=int(len(cluster_probs))
        )

    def analyze_closed_source(self, request: ClosedSourceAnalysisRequest) -> Dict:
        # Implement closed source analysis logic here
        # This would include analyzing the responses from different prompts
        # and calculating agreement scores, confidence scores, etc.
        pass

    def _get_semantic_clusters(self, embeddings: np.ndarray) -> tuple:
        if len(embeddings) < 3:
            return np.zeros(len(embeddings)), 1.0
        
        n_clusters = min(len(embeddings) // 2, 10)
        n_clusters = max(2, min(n_clusters, len(embeddings) - 1))
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='complete'
        )
        clusters = clustering.fit_predict(embeddings)
        cluster_quality = silhouette_score(embeddings, clusters, metric='cosine')
        
        return clusters, cluster_quality