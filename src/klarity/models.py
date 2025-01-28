from pydantic import BaseModel
from typing import Optional, Dict, List

class UncertaintyMetrics(BaseModel):
    raw_entropy: float
    semantic_entropy: float
    varentropy: float
    cluster_quality: float
    n_clusters: int

class UncertaintyAnalysisRequest(BaseModel):
    logits: List[float]
    prompt: str
    model_id: str
    metadata: Optional[Dict] = None

class UncertaintyAnalysisResponse(BaseModel):
    request_id: str
    metrics: UncertaintyMetrics

class ClosedSourceAnalysisRequest(BaseModel):
    prompt: str
    model_id: str
    prompt_variations: List[str]
    responses: List[str]
    validation_data: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict] = None

class PromptMetrics(BaseModel):
    weight: float
    response: str
    similarity_score: float

class ClosedSourceMetrics(BaseModel):
    prompt_metrics: List[PromptMetrics]
    agreement_score: float
    confidence_score: float
    ensemble_size: int