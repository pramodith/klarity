# basic_usage_reasoning.py
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from klarity import UncertaintyEstimator
from klarity.core.analyzer import ReasoningAnalyzer
import torch

# Initialize model with GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create estimator with reasoning analyzer and togetherai hosted model
estimator = UncertaintyEstimator(
    top_k=100,
    analyzer=ReasoningAnalyzer(
        min_token_prob=0.01,
        insight_model="together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
        insight_api_key="your_api_key",
        reasoning_start_token="<think>",
        reasoning_end_token="</think>",
    ),
)
uncertainty_processor = estimator.get_logits_processor()

# Set up generation
prompt = "Your prompt <think>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate with uncertainty analysis
generation_output = model.generate(
    **inputs,
    max_new_tokens=200,  # Increased for reasoning steps
    temperature=0.6,
    logits_processor=LogitsProcessorList([uncertainty_processor]),
    return_dict_in_generate=True,
    output_scores=True,
)

# Analyze the generation
result = estimator.analyze_generation(
    generation_output,
    tokenizer,
    uncertainty_processor,
    prompt,  # Include prompt for better reasoning analysis
)

# Get generated text
generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
print(f"\nPrompt: {prompt}")
print(f"Generated text: {generated_text}")

# Show reasoning analysis
print("\nReasoning Analysis:")
if result.overall_insight and "reasoning_analysis" in result.overall_insight:
    analysis = result.overall_insight["reasoning_analysis"]

    # Print each reasoning step
    for idx, step in enumerate(analysis["steps"], 1):
        print(f"\nStep {idx}:")  # Use simple counter instead of accessing step_number
        print(f"Content: {step['step_info']['content']}")

        # Print step analysis
        if "analysis" in step and "training_insights" in step["analysis"]:
            step_analysis = step["analysis"]["training_insights"]
            print("\nQuality Metrics:")
            for metric, score in step_analysis["step_quality"].items():
                print(f"  {metric}: {score}")

            print("\nImprovement Targets:")
            for target in step_analysis["improvement_targets"]:
                print(f"  Aspect: {target['aspect']}")
                print(f"  Importance: {target['importance']}")
                print(f"  Issue: {target['current_issue']}")
                print(f"  Suggestion: {target['training_suggestion']}")
