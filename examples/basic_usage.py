from transformers import AutoModelForCausalLM, AutoTokenizer
from klarity import UncertaintyEstimator

# Initialize
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
estimator = UncertaintyEstimator()

# Example prompt
prompt = "The capital of France is"

# Get uncertainty metrics
metrics = estimator.estimate(prompt, model, tokenizer)
print(f"Uncertainty metrics for prompt: '{prompt}'")
print(f"Raw entropy: {metrics.raw_entropy:.4f}")
print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
print(f"Variance entropy: {metrics.varentropy:.4f}")
print(f"Cluster quality: {metrics.cluster_quality:.4f}")
print(f"Number of clusters: {metrics.n_clusters}")