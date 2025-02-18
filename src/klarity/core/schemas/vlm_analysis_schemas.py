from typing import List
from pydantic import BaseModel, Field


class VLMAnalysisScoresModel(BaseModel):
    overall_uncertainty: float = Field(
        ..., description="Overall uncertainty score ranging from 0 to 1.", ge=0.0, le=1.0
    )
    visual_grounding: float = Field(
        ...,
        description="Visual grounding score indicating the quality of image-text alignment, ranging from 0 to 1.",
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        ..., description="Confidence score indicating the accuracy of the answer, ranging from 0 to 1.", ge=0.0, le=1.0
    )


class VLMAnalysisAttentionQualityModel(BaseModel):
    score: float = Field(..., description="Attention quality score ranging from 0 to 1.", ge=0.0, le=1.0)
    key_regions: List[str] = Field(..., description="Main regions of attention")
    missed_regions: List[str] = Field(..., description="Regions of attention not captured")


class VLMAnalysisTokenAttentionAlignmentModel(BaseModel):
    token: str = Field(..., description="Token")
    attended_region: str = Field(..., description="Region the token attends to")
    relevance: float = Field(..., description="Relevance score ranging from 0 to 1.", ge=0.0, le=1.0)


class VLMAnalysisVisualAnalysisModel(BaseModel):
    attention_quality: VLMAnalysisAttentionQualityModel = Field(..., description="Attention quality scores")
    token_attention_alignment: List[VLMAnalysisTokenAttentionAlignmentModel] = Field(
        ..., description="Token attention alignment"
    )


class VLMHighUncertaintySegmentsModel(BaseModel):
    text: str = Field(..., description="Text")
    reason: str = Field(..., description="Reason for high uncertainty")
    visual_context: str = Field(..., description="What the model looked at")


class VLMAnalysisImprovementSuggestions(BaseModel):
    aspect: str = Field(..., description="Aspect to improve")
    suggestion: str = Field(..., description="Suggestion for improvement")


class VLMUncertaintyAnalysisModel(BaseModel):
    high_uncertainty_segments: List[VLMHighUncertaintySegmentsModel] = Field(
        ..., description="Segments with high uncertainty"
    )
    improvement_suggestions: List[VLMAnalysisImprovementSuggestions] = Field(
        ..., description="Suggestions for improvement"
    )


class VLMVisualAnalysisModel(BaseModel):
    attention_quality: VLMAnalysisAttentionQualityModel = Field(..., description="Attention quality scores")
    token_attention_alignment: List[VLMAnalysisTokenAttentionAlignmentModel] = Field(
        ..., description="Token attention alignment"
    )


class VLMAnalysisResponseModel(BaseModel):
    scores: VLMAnalysisScoresModel = Field(..., description="Scores")
    visual_analysis: VLMVisualAnalysisModel = Field(..., description="Visual analysis")
    uncertainty_analysis: VLMUncertaintyAnalysisModel = Field(..., description="Uncertainty analysis")


class EnhancedVLMAnalysisTokenAttentionAlignmentModel(BaseModel):
    word: str = Field(..., description="The token being attended to")
    focused_spot: str = Field(..., description="Focused spot")
    relevance: float = Field(..., description="Relevance score ranging from 0 to 1.", ge=0.0, le=1.0)
    uncertainty: float = Field(..., description="Uncertainty score ranging from 0 to 1.", ge=0.0, le=1.0)


class EnhancedVLMAnalysisProblemSpotsModel(BaseModel):
    text: str = Field(..., description="Text")
    reason: str = Field(..., description="Reason for high uncertainty")
    looked_at: str = Field(..., description="What the model looked at")
    connection: str = Field(..., description="Connection between focused spot and uncertainty")


class EnhancedVLMAnalysisImprovmentTipsModel(BaseModel):
    area: str = Field(..., description="Area to improve")
    tip: str = Field(..., description="Suggestion for improvement")


class EnhancedVLMUncertaintyAnalysisModel(BaseModel):
    problem_spots: List[EnhancedVLMAnalysisProblemSpotsModel] = Field(..., description="Segments with high uncertainty")
    improvement_suggestions: List[EnhancedVLMAnalysisImprovmentTipsModel] = Field(
        ..., description="Suggestions for improvement"
    )


class EnhancedVLMVisualAnalysisModel(BaseModel):
    attention_quality: VLMAnalysisAttentionQualityModel = Field(
        ..., description="How well attention was focused for the given task"
    )
    token_attention_alignment: List[EnhancedVLMAnalysisTokenAttentionAlignmentModel] = Field(
        ..., description="Token and focused spot attention alignment"
    )


class EnhancedVLMAnalysisResponseModel(BaseModel):
    scores: VLMAnalysisScoresModel = Field(..., description="Scores")
    visual_analysis: VLMVisualAnalysisModel = Field(..., description="Visual analysis")
    uncertainty_analysis: VLMUncertaintyAnalysisModel = Field(..., description="Uncertainty analysis")
