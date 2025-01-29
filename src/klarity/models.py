from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, List

class TokenInfo(BaseModel):
    token: str
    token_id: int
    logit: float
    probability: float

class UncertaintyMetrics(BaseModel):
    raw_entropy: float
    semantic_entropy: float
    cluster_quality: Optional[float] = None  # Making optional for backward compatibility
    n_clusters: Optional[int] = None  # Making optional for backward compatibility
    coherence_score: Optional[float] = None  # New field
    divergence_score: Optional[float] = None  # New field
    hallucination_probability: Optional[float] = None  # New field
    token_predictions: List[TokenInfo]

class UncertaintyAnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    logits: List[float]
    prompt: str
    model_id: str
    token_info: List[TokenInfo]
    metadata: Optional[Dict] = None

class UncertaintyAnalysisResponse(BaseModel):
    request_id: str
    metrics: UncertaintyMetrics

class PromptMetrics(BaseModel):
    weight: float
    response: str
    similarity_score: float