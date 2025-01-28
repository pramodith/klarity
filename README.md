<div align="center">
  <img src="assets/klaralabs.png" alt="Klara Labs" width="200"/>
  <br>
  <br>
  <img src="assets/detectivebird.jpeg" alt="Mascotte" width="200" style="border-radius: 20px; margin: 20px 0;"/>

  # Klarity 

  _Understand what your model is thinking_
</div>

## 🎯 Overview

Klarity is a powerful tool for understanding the semantic uncertainty in language model outputs. Unlike traditional confidence scores, Klarity analyzes the semantic space of possible responses to provide deep insights into your model's uncertainty.

### What sets Klarity apart?

- **Semantic Analysis**: Goes beyond raw probability distributions to understand the meaning-based uncertainty in model outputs
- **Cluster Insights**: Identifies semantic groupings in the model's potential responses
- **Quality Metrics**: Provides comprehensive metrics about response coherence and diversity

## 🔍 Semantic Uncertainty Insights

Klarity provides several key metrics:

| Metric | Description | Use Case |
|--------|-------------|----------|
| Semantic Entropy | Measures uncertainty in the meaning space | Identify when your model is considering semantically different responses |
| Cluster Quality | Evaluates the coherence of response clusters | Detect when responses are well-organized vs scattered |
| Variance Entropy | Quantifies the stability of model responses | Monitor response consistency over multiple runs |

## 🚀 Quick Start

Install directly from GitHub:
```bash
pip install git+https://github.com/yourusername/klarity.git
```

Start analyzing your model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from klarity import UncertaintyEstimator

# Load your model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize Klarity
estimator = UncertaintyEstimator()

# Get semantic insights
metrics = estimator.estimate("What is the best programming language?", model, tokenizer)

# Analyze the results
print(f"Semantic Entropy: {metrics.semantic_entropy:.4f}")
print(f"Cluster Quality: {metrics.cluster_quality:.4f}")
print(f"Number of Semantic Clusters: {metrics.n_clusters}")
```

## 🤖 Supported Models

| Model | Size | Status | Use Case |
|-------|-------|--------|-----------|
| Llama 3 | 8B | ✅ Tested | Great for general text generation analysis |
| Qwen | 0.5B | ✅ Tested | Efficient for quick uncertainty assessments |

> **Note**: More models are being tested! If you'd like to contribute by testing additional models, please submit a PR.

## 💡 Example Use Cases

### Model Understanding
- Identify when your model is uncertain between distinct semantic concepts
- Understand the coherence of your model's semantic space
- Track how uncertainty changes with different prompts

### Quality Assurance
- Detect edge cases where your model considers multiple valid but different responses
- Monitor semantic stability across different versions of your model
- Validate response quality through semantic clustering

### Model Improvement
- Use semantic insights to guide model fine-tuning
- Identify areas where your model needs more diverse training data
- Optimize for semantic coherence in responses

## 📚 Documentation

Detailed documentation is available in our [Wiki](https://github.com/yourusername/klarity/wiki), including:
- Semantic Metrics Guide
- Implementation Examples
- Advanced Configuration
- Best Practices

## 🤝 Contributing

We welcome contributions! Whether it's:
- Testing with new models
- Adding new semantic analysis features
- Improving documentation
- Sharing use cases

See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

MIT License. See [LICENSE](LICENSE) for more information.

## 📫 Community & Support

- [GitHub Issues](https://github.com/yourusername/klarity/issues) for bugs and features
- [GitHub Discussions](https://github.com/yourusername/klarity/discussions) for questions and community discussions