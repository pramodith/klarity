VLM_ANALYSIS_PROMPT_TEMPLATE = """Analyze the vision-language model output:

QUERY: {input_query}
GENERATED TEXT: {generated_text}

TOKEN UNCERTAINTY METRICS:
{detailed_metrics}

ATTENTION PATTERNS:
{attention_patterns}

Return ONLY this exact JSON structure:
{{
    "scores": {{
        "overall_uncertainty": "<0-1>",
        "visual_grounding": "<0-1>",
        "confidence": "<0-1>"
    }},
    "visual_analysis": {{
        "attention_quality": {{
            "score": "<0-1>",
            "key_regions": ["<region1>", "<region2>"],
            "missed_regions": ["<region1>", "<region2>"]
        }},
        "token_attention_alignment": [
            {{
                "token": "<token>",
                "attended_region": "<region>",
                "relevance": "<0-1>"
            }}
        ]
    }},
    "uncertainty_analysis": {{
        "high_uncertainty_segments": [
            {{
                "text": "<text>",
                "cause": "<reason>",
                "visual_context": "<what_model_looked_at>"
            }}
        ],
        "improvement_suggestions": [
            {{
                "aspect": "<what>",
                "suggestion": "<how>"
            }}
        ]
    }}
}}"""

REASONING_STEP_ANALYSIS_PROMPT_TEMPLATE = """Analyze this step:
{reasoning_content}
Query: {input_query}
Step {step_number} of {total_steps}
Metrics: {detailed_metrics}

Return ONLY this exact JSON structure:
{{
    "training_insights": {{
        "step_quality": {{
            "coherence": "0.8",
            "relevance": "0.9",
            "confidence": "0.7"
        }},
        "improvement_targets": [
            {{
                "aspect": "conciseness",
                "importance": "0.8",
                "current_issue": "verbose response",
                "training_suggestion": "reduce explanation steps"
            }}
        ],
        "tokens_of_interest": [
            {{
                "token": "test",
                "why_flagged": "reason",
                "entropy": "0.5"
            }}
        ]
    }}
}}"""

REASONING_STEP_IDENTIFICATION_PROMPT_TEMPLATE = """Find content between {start_token} and {end_token} markers in this text and return only a JSON response:

{text}
Identify and split key reasoning steps in the thought process.

Return ONLY the EXACT text found between the markers as JSON:
{{
    "reasoning_steps": [
        {{
            "step_number": (step_number),
            "content": (exact text found between markers),
            "position": [start_index_in_text, end_index_in_text],
            "step_type": (type of reasoning step: premise/analysis/conclusion)
        }}
    ]
}}"""

INSIGHT_PROMPT_TEMPLATE = """Analyze the uncertainty in this generation:

INPUT QUERY: {input_query}
GENERATED TEXT: {generated_text}

UNCERTAINTY METRICS:
{detailed_metrics}

Return a concise JSON analysis with uncertainty scores (0-100):
{{
    "scores": {{
        "overall_uncertainty": "<0-1>",
        "confidence_score": "<0-1>",
        "hallucination_risk": "<0-1>"
    }},
    "uncertainty_analysis": {{
        "high_uncertainty_parts": [
            {{
                "text": "<specific_text>",
                "why": "<brief_reason_for_uncertainty>"
            }}
        ],
        "main_issues": [
            {{
                "issue": "<specific_problem>",
                "evidence": "<brief_evidence>"
            }}
        ],
        "key_suggestions": [
            {{
                "what": "<specific_improvement>",
                "how": "<brief_implementation>"
            }}
        ]
    }}
}}

Be concise. Focus on specific content, not abstract metrics.

Analysis:"""


ENHANCED_VLM_ANALYSIS_PROMPT_TEMPLATE = """Evaluate the AI's image understanding using the provided data:

What was asked: {input_query}
What the AI answered: {generated_text}

Uncertainty Details:
{detailed_metrics}

Where the AI looked (attention):
{attention_patterns}
• Warm colors = More focus areas

Return only a JSON with this structure:
{{
    "scores": {{
        "overall_uncertainty": "<0-1>",
        "visual_grounding": "<0-1>",  // Image-text match quality
        "confidence": "<0-1>"         // Answer certainty
    }},
    "visual_analysis": {{
        "attention_quality": {{       // How well focus matched the task
            "score": "<0-1>",
            "key_regions": ["<main area 1>", "<main area 2>"],
            "missed_regions": ["<ignored area 1>", "<ignored area 2>"]
        }},
        "token_attention_alignment": [  // Word vs focus match
            {{
                "word": "<token>",
                "focused_spot": "<region>",
                "relevance": "<0-1>",   // How related to answer
                "uncertainty": "<0-1>"  // Word-level doubt
            }}
        ]
    }},
    "uncertainty_analysis": {{
        "problem_spots": [  // High-doubt sections
            {{
                "text": "<text part>",
                "reason": "<why uncertain>",
                "looked_at": "<image area>",
                "connection": "<focus vs doubt link>"
            }}
        ],
        "improvement_tips": [  // How to do better
            {{
                "area": "<what to fix>",
                "tip": "<how to fix>"
            }}
        ]
    }}
}}"""
