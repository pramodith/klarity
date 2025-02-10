# basic_usage_together_ai.py
from klarity import UncertaintyEstimator
from klarity.core.analyzer import EntropyAnalyzer

# Initialize for Together AI model
estimator = UncertaintyEstimator(
    top_k=5,
    analyzer=EntropyAnalyzer(
        min_token_prob=0.01,
        insight_model="together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
        insight_api_key="your_api_key",
    ),
    together_api_key="your_api_key",
    together_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

# Generate and analyze
prompt = "Your prompt"
generation_output = estimator._generate_with_together(prompt=prompt, max_new_tokens=10, temperature=0.7)

result = estimator.analyze_generation(generation_output)

# Print results
print(f"\nPrompt: {prompt}")
print(f"Generated text: {generation_output['text']}")

print("\nDetailed Token Analysis:")
for idx, metrics in enumerate(result.token_metrics):
    print(f"\nStep {idx}:")
    print(f"Raw entropy: {metrics.raw_entropy:.4f}")
    print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
    print("Top 3 predictions:")
    for i, pred in enumerate(metrics.token_predictions[:3], 1):
        print(f"  {i}. {pred.token} (prob: {pred.probability:.4f})")

if result.overall_insight:
    print("\nComprehensive Analysis:")
    print(result.overall_insight)
