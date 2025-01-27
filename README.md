<div align="center">
  <img src="assets/klaralabs.png" alt="Klara Labs" width="200"/>

  # Klarity 

  **See Through Your Models**
</div>

<div align="center">
  <img src="assets/klaralabs.png" alt="Klara Labs" width="200"/>

  # Klarity 

  **See Through Your Models**
</div>

## 🤖 Supported Models

| Model | Size | Status | Notes |
|-------|-------|--------|--------|
| Llama 2 | 8B | ✅ Tested | Full compatibility with uncertainty metrics |
| Qwen | 0.5B | ✅ Tested | Full compatibility with uncertainty metrics |

> **Note**: More models are being tested and will be added to the list. If you'd like to contribute by testing additional models, please submit a PR!


## 🚀 Installation

1. Install Poetry (our package manager):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install Klarity and activate the environment:
```bash
poetry install
poetry shell
```

## 🔧 Getting Started

1. First, set up your environment with your API key:

```python
import os
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Set your Klara Labs API key
os.environ['KLARALABS_API_KEY'] = "your_api_key_here"
```

2. Track uncertainty in your models:

```python
from klaralabs.uncertainty import KlaraLabsUncertainty
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize KlaraLabs
kl = KlaraLabsUncertainty.cloud(verbose=True)

# Load your model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Estimate uncertainty
uncertainty = kl.estimate("What is 1+1?", model, tokenizer)
# Access metrics like semantic_entropy, raw_entropy, cluster_quality
```

## 📊 Features

- Model uncertainty estimation
- Semantic entropy analysis
- Raw entropy calculation
- Cluster quality assessment
- Real-time monitoring of uncertainty metrics
- Dashboard for visualization and tracking

## 💡 Use Cases

- Monitor your model's uncertainty metrics
- Analyze potential failure modes
- Track uncertainty patterns over time
- Improve model reliability
- Identify edge cases and anomalies

## 📝 License

[Add your license information here]

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📫 Support

For support and questions, please [open an issue](https://github.com/yourusername/klarity/issues) or contact us at [your contact information].