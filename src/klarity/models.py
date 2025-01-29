# models.py
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, List

class UncertaintyMetrics(BaseModel):
    raw_entropy: float
    semantic_entropy: float
    cluster_quality: float
    n_clusters: int

class UncertaintyAnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    logits: List[float]
    prompt: str
    model_id: str
    metadata: Optional[Dict] = None

class UncertaintyAnalysisResponse(BaseModel):
    request_id: str
    metrics: UncertaintyMetrics

class PromptMetrics(BaseModel):
    weight: float
    response: str
    similarity_score: float