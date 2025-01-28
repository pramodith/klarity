<div align="center">
  <img src="assets/klaralabs.png" alt="Klara Labs" width="200"/>
  <br>
  <br>
  <img src="assets/detectivebird.jpeg" alt="Mascotte" width="200" style="border-radius: 20px; margin: 20px 0;"/>

  # Klarity 

  _Understand what your model is thinking_
</div>


## 🤖 Supported Models

| Model | Size | Status | Notes |
|-------|-------|--------|--------|
| Llama 3 | 8B | ✅ Tested | Full compatibility with uncertainty metrics |
| Qwen | 0.5B | ✅ Tested | Full compatibility with uncertainty metrics |

> **Note**: More models are being tested and will be added to the list. If you'd like to contribute by testing additional models, please submit a PR!


## 🚀 Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/yourusername/klarity.git
```

Optional: For development, clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/klarity.git
cd klarity
pip install -e .
```

## 🔧 Getting Started

Here's how to get started with Klarity:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from klarity import UncertaintyEstimator

# Load your model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the uncertainty estimator
estimator = UncertaintyEstimator()

# Estimate uncertainty
metrics = estimator.estimate("What is 1+1?", model, tokenizer)

# Access the metrics
print(f"Raw entropy: {metrics.raw_entropy:.4f}")
print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
print(f"Variance entropy: {metrics.varentropy:.4f}")
print(f"Cluster quality: {metrics.cluster_quality:.4f}")
print(f"Number of clusters: {metrics.n_clusters}")
```

## 📊 Features

- Model uncertainty estimation through entropy analysis
- Semantic clustering of token distributions
- Multiple uncertainty metrics:
  - Raw entropy for direct uncertainty measurement
  - Semantic entropy for meaning-based uncertainty
  - Variance entropy for response stability
  - Cluster quality assessment
  - Semantic cluster analysis

## 💡 Use Cases

- Monitor your model's uncertainty metrics
- Analyze potential failure modes
- Track uncertainty patterns over time
- Improve model reliability
- Identify edge cases and anomalies

## 📝 License

MIT License. See [LICENSE](LICENSE) for more information.

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📫 Support

- For bugs and feature requests, please [open an issue](https://github.com/yourusername/klarity/issues)
- For questions and discussions, feel free to start a [GitHub Discussion](https://github.com/yourusername/klarity/discussions)