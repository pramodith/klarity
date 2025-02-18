from typing import List

from pydantic import BaseModel, Field


class Scores(BaseModel):
    overall_uncertainty: float = Field(
        ..., description="Overall uncertainty score indicating the model's general uncertainty level in the range [0, 1]"
    )
    confidence_score: float = Field(..., description="Model's confidence in its analysis and predictions in the range [0, 1]")
    hallucination_risk: float = Field(
        ..., description="Risk level of model hallucination or fabrication in the response in the range [0, 1]"
    )


class UncertaintyPart(BaseModel):
    text: str = Field(..., description="Specific text segment that has high uncertainty")
    why: str = Field(..., description="Brief explanation of why this part has high uncertainty")


class Issue(BaseModel):
    issue: str = Field(..., description="Specific problem or issue identified in the analysis")
    evidence: str = Field(..., description="Brief evidence or context supporting the identified issue")


class Suggestion(BaseModel):
    what: str = Field(..., description="Specific improvement or change suggested")
    how: str = Field(..., description="Brief implementation details or steps for the suggestion")


class UncertaintyAnalysis(BaseModel):
    high_uncertainty_parts: List[UncertaintyPart] = Field(
        ..., description="List of text segments with high uncertainty and their explanations"
    )
    main_issues: List[Issue] = Field(
        ..., description="List of main issues identified in the analysis with supporting evidence"
    )
    key_suggestions: List[Suggestion] = Field(
        ..., description="List of key suggestions for improvement with implementation details"
    )


class InsightAnalysisResponseModel(BaseModel):
    scores: Scores = Field(
        ..., description="Collection of various analysis scores including uncertainty and confidence"
    )
    uncertainty_analysis: UncertaintyAnalysis = Field(
        ..., description="Detailed analysis of uncertainty including issues and suggestions"
    )