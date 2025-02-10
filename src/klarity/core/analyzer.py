# analyzer.py
from typing import Any, List, Dict, Optional
from klarity.core.together_wrapper import TogetherModelWrapper
import numpy as np
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from ..models import TokenInfo, UncertaintyAnalysisRequest, UncertaintyMetrics
import json
import re
import traceback


class EntropyAnalyzer:
    def __init__(
        self,
        min_token_prob: float = 0.01,
        insight_model: Optional[Any] = None,
        insight_tokenizer: Optional[Any] = None,
        insight_api_key: Optional[str] = None,
        insight_prompt_template: Optional[str] = None,
    ):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.min_token_prob = min_token_prob

        # Initialize Together AI model if specified
        self.together_model = None
        self.insight_model = None
        self.insight_tokenizer = None

        if isinstance(insight_model, str) and insight_model.startswith("together:"):
            model_name = insight_model.replace("together:", "")
            self.together_model = TogetherModelWrapper(
                model_name=model_name, api_key=insight_api_key
            )
        else:
            self.insight_model = insight_model
            self.insight_tokenizer = insight_tokenizer

        self.insight_prompt_template = (
            insight_prompt_template
            or """Analyze the uncertainty in this generation:

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
        )

    def _calculate_raw_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate raw entropy of probability distribution"""
        max_entropy = np.log(len(probabilities))
        raw_entropy = entropy(probabilities)
        return raw_entropy / max_entropy if max_entropy != 0 else 0.0

    def _calculate_semantic_entropy(self, token_info: List[TokenInfo]) -> float:
        """Calculate semantic entropy based on token predictions"""
        if len(token_info) < 2:
            return 0.0

        tokens = [t.token for t in token_info]
        embeddings = self.embedding_model.encode(tokens)
        similarity_matrix = cosine_similarity(embeddings)
        semantic_groups = self._group_similar_tokens(similarity_matrix, token_info)
        group_probs = self._calculate_group_probabilities(semantic_groups, token_info)

        if len(group_probs) > 1:
            return entropy(list(group_probs.values())) / np.log(len(group_probs))
        return 0.0

    def _group_similar_tokens(
        self,
        similarity_matrix: np.ndarray,
        token_info: List[TokenInfo],
        threshold: float = 0.8,
    ) -> Dict[int, List[int]]:
        """Group tokens with similar semantic meanings"""
        groups = defaultdict(list)
        group_id = 0

        for i in range(len(token_info)):
            assigned = False
            for gid in groups:
                if any(similarity_matrix[i][j] > threshold for j in groups[gid]):
                    groups[gid].append(i)
                    assigned = True
                    break
            if not assigned:
                groups[group_id] = [i]
                group_id += 1

        return groups

    def _calculate_group_probabilities(
        self, semantic_groups: Dict[int, List[int]], token_info: List[TokenInfo]
    ) -> Dict[int, float]:
        """Calculate probability mass for each semantic group"""
        group_probs = defaultdict(float)
        total_prob = 0.0

        for gid, indices in semantic_groups.items():
            group_prob = sum(token_info[i].probability for i in indices)
            group_probs[gid] = group_prob
            total_prob += group_prob

        if total_prob > 0:
            for gid in group_probs:
                group_probs[gid] /= total_prob

        return group_probs

    def generate_overall_insight(
        self,
        metrics_list: List[UncertaintyMetrics],
        input_query: Optional[str] = "",
        generated_text: Optional[str] = "",
    ) -> Optional[str]:
        """Generate overall insight for all collected metrics"""
        if not self.together_model and not (
            self.insight_model and self.insight_tokenizer
        ):
            return None

        # Format metrics for the prompt
        detailed_metrics = []
        for idx, metrics in enumerate(metrics_list):
            top_predictions = [
                f"{t.token} ({t.probability:.3f})"
                for t in metrics.token_predictions[:3]
            ]

            step_metrics = (
                f"Step {idx}:\n"
                f"- Raw Entropy: {metrics.raw_entropy:.4f}\n"
                f"- Semantic Entropy: {metrics.semantic_entropy:.4f}\n"
                f"- Top 3 Predictions: {' | '.join(top_predictions)}"
            )
            detailed_metrics.append(step_metrics)

        all_metrics = "\n\n".join(detailed_metrics)
        prompt = self.insight_prompt_template.format(
            detailed_metrics=all_metrics,
            input_query=input_query,
            generated_text=generated_text,
        )

        if self.together_model:
            return self.together_model.generate_insight(prompt)

        # Use original model
        inputs = self.insight_tokenizer(prompt, return_tensors="pt").to(
            self.insight_model.device
        )
        outputs = self.insight_model.generate(
            inputs.input_ids,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        return (
            self.insight_tokenizer.decode(outputs[0], skip_special_tokens=True)
            .split("Analysis:")[-1]
            .strip()
        )

    def analyze(self, request: UncertaintyAnalysisRequest) -> UncertaintyMetrics:
        """Analyze uncertainty for a single generation step"""
        probabilities = np.array([t.probability for t in request.token_info])
        raw_entropy = self._calculate_raw_entropy(probabilities)
        semantic_entropy = self._calculate_semantic_entropy(request.token_info)

        metrics = UncertaintyMetrics(
            raw_entropy=float(raw_entropy),
            semantic_entropy=float(semantic_entropy),
            token_predictions=request.token_info,
        )

        return metrics


class ReasoningAnalyzer(EntropyAnalyzer):
    def __init__(
        self,
        reasoning_start_token: str = "<think>",
        reasoning_end_token: str = "</think>",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reasoning_start_token = reasoning_start_token
        self.reasoning_end_token = reasoning_end_token

        # Template to identify reasoning steps
        self.reasoning_identification_template = """Find content between {start_token} and {end_token} markers in this text and return only a JSON response:

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

        self.step_analysis_template = """Analyze this step:
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

    def identify_reasoning_steps(self, text: str) -> List[Dict]:
        """Use the insight model to identify reasoning steps"""
        try:
            prompt = self.reasoning_identification_template.format(
                start_token=self.reasoning_start_token,
                end_token=self.reasoning_end_token,
                text=text,
            )

            if self.together_model:
                response = self.together_model.generate_insight(prompt)
                print("\nDEBUG - Raw response:")
                print(response)

                # Clean up the response by removing markdown code blocks
                cleaned_response = (
                    response.replace("```json", "").replace("```", "").strip()
                )

                try:
                    parsed = json.loads(cleaned_response)
                    print("\nDEBUG - Parsed JSON:")
                    print(json.dumps(parsed, indent=2))
                    return parsed.get("reasoning_steps", [])
                except json.JSONDecodeError as e:
                    print(f"\nDEBUG - JSON parsing error: {e}")
                    print("Cleaned response:", cleaned_response)
                    return []

        except Exception as e:
            print(f"Error in identify_reasoning_steps: {str(e)}")
            return []

    def analyze_reasoning_step(
        self,
        step_info: Dict,
        metrics_list: List[UncertaintyMetrics],
        input_query: str,
        total_steps: int,
    ) -> Dict:
        try:
            start_idx = step_info["position"][0]
            end_idx = step_info["position"][1]
            step_metrics = self._get_metrics_for_range(metrics_list, start_idx, end_idx)

            detailed_metrics = self._format_metrics(step_metrics)

            prompt = self.step_analysis_template.format(
                reasoning_content=step_info["content"],
                input_query=input_query,
                step_number=step_info.get("step_number", 1),
                total_steps=total_steps,
                detailed_metrics=detailed_metrics,
            )

            if self.together_model:
                response = self.together_model.generate_insight(prompt)
                print(f"\nDEBUG - Raw response from analysis: {response}")

                # Find JSON content
                start = response.find("{")
                end = response.rfind("}") + 1

                if start >= 0 and end > start:
                    json_str = response[start:end]
                    # Remove any potential line breaks or comments in the JSON
                    json_str = re.sub(r"#.*$", "", json_str, flags=re.MULTILINE)
                    json_str = json_str.replace("\n", "").replace("\\n", "")

                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {e}")
                        print(f"Attempted to parse: {json_str}")
                        # Return a default structure
                        return {
                            "training_insights": {
                                "step_quality": {
                                    "coherence": "0.5",
                                    "relevance": "0.5",
                                    "confidence": "0.5",
                                },
                                "improvement_targets": [],
                                "tokens_of_interest": [],
                            }
                        }

                return {"error": "No JSON found in response"}

        except Exception as e:
            print(f"Error in analyze_reasoning_step: {str(e)}")
            traceback.print_exc()  # Add this for more detailed error info
            return {"error": f"Analysis failed: {str(e)}"}

    def _get_metrics_for_range(
        self, metrics_list: List[UncertaintyMetrics], start_idx: int, end_idx: int
    ) -> List[UncertaintyMetrics]:
        """Extract metrics for a specific token range"""
        return metrics_list[start_idx:end_idx]

    def _format_metrics(self, metrics: List[UncertaintyMetrics]) -> str:
        """Format metrics for the prompt"""
        formatted = []
        for idx, metric in enumerate(metrics):
            top_predictions = [
                f"{t.token} ({t.probability:.3f})" for t in metric.token_predictions[:3]
            ]

            metric_text = (
                f"Token {idx}:\n"
                f"- Raw Entropy: {metric.raw_entropy:.4f}\n"
                f"- Semantic Entropy: {metric.semantic_entropy:.4f}\n"
                f"- Top Predictions: {' | '.join(top_predictions)}"
            )
            formatted.append(metric_text)

        return "\n\n".join(formatted)

    def generate_overall_insight(
        self,
        metrics_list: List[UncertaintyMetrics],
        input_query: Optional[str] = "",
        generated_text: Optional[str] = "",
    ) -> Optional[Dict]:
        """Generate comprehensive analysis of all reasoning steps"""
        if not (self.together_model or (self.insight_model and self.insight_tokenizer)):
            return None

        # Identify reasoning steps
        reasoning_steps = self.identify_reasoning_steps(generated_text)
        if not reasoning_steps:
            return None

        # Analyze each step
        step_analyses = []
        for step in reasoning_steps:
            analysis = self.analyze_reasoning_step(
                step, metrics_list, input_query, len(reasoning_steps)
            )
            step_analyses.append({"step_info": step, "analysis": analysis})

        # Compile overall assessment
        return {
            "reasoning_analysis": {
                "steps": step_analyses,
                "overall_metrics": {
                    "total_steps": len(reasoning_steps),
                    "average_raw_entropy": sum(m.raw_entropy for m in metrics_list)
                    / len(metrics_list),
                    "average_semantic_entropy": sum(
                        m.semantic_entropy for m in metrics_list
                    )
                    / len(metrics_list),
                    "reasoning_flow_score": self._calculate_flow_score(step_analyses),
                },
            }
        }

    def _calculate_flow_score(self, step_analyses: List[Dict]) -> float:
        """Calculate how well reasoning steps flow together"""
        try:
            # Update this to match the actual JSON structure we receive
            coherence_scores = [
                float(
                    step["analysis"]
                    .get("training_insights", {})
                    .get("step_quality", {})
                    .get("coherence", "0.5")
                )
                for step in step_analyses
            ]
            return (
                sum(coherence_scores) / len(coherence_scores)
                if coherence_scores
                else 0.0
            )
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Error calculating flow score: {e}")
            return 0.0
