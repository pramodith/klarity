#basic_usage.py
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from klarity import UncertaintyEstimator
from klarity.core.analyzer import EntropyAnalyzer

# Initialize
model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create estimator
estimator = UncertaintyEstimator(
    top_k=100,
    analyzer=EntropyAnalyzer(min_token_prob=0.01)
)
uncertainty_processor = estimator.get_logits_processor()

# Set up generation
prompt = "What is the capital of furufurufru?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with uncertainty analysis
generation_output = model.generate(
    **inputs,
    max_new_tokens=20,
    temperature=0.7,
    top_p=0.9,
    logits_processor=LogitsProcessorList([uncertainty_processor]),
    return_dict_in_generate=True,
    output_scores=True,
)

# Analyze the generation
metrics_list = estimator.analyze_generation(
    generation_output,
    tokenizer,
    uncertainty_processor
)

# Get generated text
generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
print(f"\nPrompt: {prompt}")
print(f"Generated text: {generated_text}")

# Print token-level metrics
print("\nToken-level Analysis:")
for i, metrics in enumerate(metrics_list):
    print(f"\nToken {i}:")
    print(f"Raw entropy: {metrics.raw_entropy:.4f}")
    print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
    print("Top 3 predictions:")
    for pred in metrics.token_predictions[:3]:
        print(f"  {pred.token}: {pred.probability:.3f}")