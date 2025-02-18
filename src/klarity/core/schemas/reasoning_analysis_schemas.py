from enum import Enum

from pydantic import BaseModel, Field


class StepType(Enum):
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"
    PREMISE = "premise"


class ReasoningStepIdentificationResponseModel(BaseModel):
    step_number: int = Field(..., description="Number of the reasoning step")
    content: str = Field(..., description="Exact text found between the start and end tokens")
    position: list[int] = Field(
        ..., description="Position of the start and end tokens corresponding to the reasoning step"
    )
    step_type: StepType = Field(..., description="Type of reasoning step")


class ReasoningStepQualityModel(BaseModel):
    coherence: float = Field(..., description="Coherence score ranging from 0 to 1.", ge=0.0, le=1.0)
    relevance: float = Field(..., description="Relevance score ranging from 0 to 1.", ge=0.0, le=1.0)
    confidence: float = Field(..., description="Confidence score ranging from 0 to 1.", ge=0.0, le=1.0)


class ReasoningStepImprovementTargetModel(BaseModel):
    aspect: str = Field(..., description="Aspect to improve")
    importance: float = Field(..., description="Importance score ranging from 0 to 1.", ge=0.0, le=1.0)
    current_issue: str = Field(..., description="Current issue that needs improvement")
    training_suggestion: str = Field(..., description="Training suggestion")


class ReasoningStepTokensOfInterestModel(BaseModel):
    token: str = Field(..., description="Token")
    why_flagged: str = Field(..., description="Why the token was flagged")
    entropy: float = Field(..., description="Entropy of the token")


class ReasoningStepAnalysisTrainingInsightsModel(BaseModel):
    step_quality: ReasoningStepQualityModel = Field(..., description="Quality metrics for the reasoning step")
    improvement_targets: list[ReasoningStepImprovementTargetModel] = Field(
        ..., description="Areas for improvement based on the reasoning step"
    )
    tokens_of_interest: list[ReasoningStepTokensOfInterestModel] = Field(
        ..., description="Tokens that may be relevant to the reasoning step"
    )


class ReasoningStepAnalysisResponseModel(BaseModel):
    training_insights: ReasoningStepAnalysisTrainingInsightsModel = Field(
        ..., description="Training insights for the reasoning step"
    )
    improvement_targets: list[ReasoningStepImprovementTargetModel] = Field(
        ..., description="Areas for improvement based on the reasoning step"
    )
    tokens_of_interest: list[ReasoningStepTokensOfInterestModel] = Field(
        ..., description="Tokens that may be relevant to the reasoning step"
    )
