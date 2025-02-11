# models.py
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, List, Union, Any


class TokenInfo(BaseModel):
    token: str
    token_id: int
    logit: float
    probability: float
    attention_score: Optional[float] = None  # For VLM token-specific attention


class AttentionData(BaseModel):
    """Model for VLM attention patterns"""

    cumulative_attention: Optional[Any] = None  # numpy array of attention weights
    token_attentions: Optional[List[Dict[str, Any]]] = None  # List of per-token attention data


class UncertaintyMetrics(BaseModel):
    raw_entropy: float
    semantic_entropy: float
    token_predictions: List[TokenInfo]
    insight: Optional[str] = None
    attention_metrics: Optional[Dict[str, float]] = None  # For VLM attention metrics


class UncertaintyAnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    logits: Optional[List[float]] = None  # Made optional for VLM cases
    prompt: str
    model_id: str
    token_info: List[TokenInfo]
    metadata: Optional[Dict] = None
    attention_maps: Optional[Any] = None  # For VLM attention maps


class UncertaintyAnalysisResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    token_metrics: List[UncertaintyMetrics]
    overall_insight: Optional[Union[str, Dict]] = None
    attention_data: Optional[AttentionData] = None  # For VLM attention analysis
