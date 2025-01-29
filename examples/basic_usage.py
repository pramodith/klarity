from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from klarity import UncertaintyEstimator
from klarity.core.analyzer import EntropyAnalyzer  # Import the new analyzer

# Initialize
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create estimator with enhanced analyzer
estimator = UncertaintyEstimator(
    top_k=100,
    analyzer=EntropyAnalyzer(
        window_size=5,  # Size of sliding window for context
        min_token_prob=0.01  # Minimum probability threshold for tokens
    )
)
uncertainty_processor = estimator.get_logits_processor()

# Set up generation
prompt = "What is the color of the red hat?"
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

# Print enhanced metrics
print("\nUncertainty Analysis:")
for i, metrics in enumerate(metrics_list):
    print(f"\nWindow {i}:")
    print(f"Raw entropy: {metrics.raw_entropy:.4f}")
    print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
    print(f"Coherence score: {metrics.coherence_score:.4f}")
    print(f"Divergence score: {metrics.divergence_score:.4f}")
    print(f"Hallucination probability: {metrics.hallucination_probability:.4f}")
    print("Top 3 predictions:")
    for pred in metrics.token_predictions[:3]:
        print(f"  {pred.token}: {pred.probability:.3f}")

# Optional: Print overall analysis
if metrics_list:
    avg_hallucination_prob = sum(m.hallucination_probability for m in metrics_list) / len(metrics_list)
    avg_coherence = sum(m.coherence_score for m in metrics_list) / len(metrics_list)
    print(f"\nOverall Analysis:")
    print(f"Average hallucination probability: {avg_hallucination_prob:.4f}")
    print(f"Average coherence score: {avg_coherence:.4f}")