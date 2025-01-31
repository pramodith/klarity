# Klarity

<div align="center">
  <img src="assets/klaralabs.png" alt="Klara Labs" width="200"/>
  <br>
  <br>
  <img src="assets/detectivebird.jpeg" alt="Mascotte" width="200" style="border-radius: 20px; margin: 20px 0;"/>

  # Klarity 

  _Understanding uncertainty in generative model predictions_
</div>

## 🎯 Overview

Klarity is a tool for analyzing uncertainty in generative model outputs. It combines both raw probability analysis and semantic understanding to provide deep insights into model behavior during text generation. The library offers:

- **Dual Entropy Analysis**: Combines raw probability entropy with semantic similarity-based entropy
- **Semantic Clustering**: Groups similar predictions to understand model decision-making
- **AI-Powered Insights**: Uses a separate model to analyze generation patterns and provide human-readable insights
- **Step-by-Step Analysis**: Tracks uncertainty metrics at each generation step

<div align="center">
  <br>

  <img src="assets/example.png" alt="example" width="400"/>
  
  <br>

</div>

## 🚀 Quick Start

Install directly from GitHub:
```bash
pip install git+https://github.com/klaralabs/klarity.git
```

Basic usage example:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from klarity import UncertaintyEstimator
from klarity.core.analyzer import EntropyAnalyzer

# Initialize models
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Initialize insight model
insight_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
insight_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Create estimator
estimator = UncertaintyEstimator(
    top_k=100,
    analyzer=EntropyAnalyzer(
        min_token_prob=0.01,
        insight_model=insight_model,
        insight_tokenizer=insight_tokenizer
    )
)

# Set up generation with uncertainty analysis
uncertainty_processor = estimator.get_logits_processor()
prompt = "What is the capital of Italy?"

# Generate with analysis
inputs = tokenizer(prompt, return_tensors="pt")
generation_output = model.generate(
    **inputs,
    max_new_tokens=20,
    temperature=0.7,
    top_p=0.9,
    logits_processor=LogitsProcessorList([uncertainty_processor]),
    return_dict_in_generate=True,
    output_scores=True,
)

# Get analysis results
result = estimator.analyze_generation(
    generation_output,
    tokenizer,
    uncertainty_processor
)
```

## 🦉 Klara Labs API

For users seeking more powerful insight generation without managing infrastructure, we offer a hosted API solution:

- **Enhanced Models**: Access to state-of-the-art models optimized for explainability
- **Higher Accuracy**: Improved semantic analysis and clustering
- **Managed Infrastructure**: No need to handle model deployment and scaling
- **Simple Integration**: RESTful API with comprehensive documentation

Visit [klaralabs.com](https://klaralabs.com) to learn more about our platform.

## 📊 Analysis Metrics

Klarity provides several key metrics for each generation step:

- **Raw Entropy**: Traditional entropy based on token probabilities
- **Semantic Entropy**: Entropy calculated over semantically similar token groups
- **Token Predictions**: Detailed probability distribution of top predicted tokens
- **Overall Insight**: AI-generated explaination of uncertainty patterns

## 🤖 Supported Frameworks & Models

### Model Frameworks
Currently supported:
- ✅ Hugging Face Transformers

Planned support:
- ⏳ PyTorch

### Tested Models
| Model | Type | Status | Notes |
|-------|------|--------|--------|
| Qwen2.5-0.5B | Base | ✅ Tested | Full Support |
| Qwen2.5-0.5B-Instruct | Instruct | ✅ Tested | Full Support |

### Insight Models
| Framework | Status | Notes |
|-----------|--------|--------|
| Hugging Face | ✅ Supported | Requires Causal LM |

## 🔍 Advanced Features

### Custom Insight Templates
You can customize the insight generation prompt:
```python
custom_template = """
Analyze metrics:
{detailed_metrics}

Provide insights on:
1. Uncertainty patterns
2. Decision points
3. Recommendations

Analysis:
"""

analyzer = EntropyAnalyzer(
    insight_prompt_template=custom_template,
    insight_model=model,
    insight_tokenizer=tokenizer
)
```

### Semantic Analysis Configuration
```python
analyzer = EntropyAnalyzer(
    min_token_prob=0.01,  # Minimum probability threshold
)
```

## 🤝 Contributing

Contributions are welcome! Areas we're looking to improve:

- Additional framework support
- More tested models
- Enhanced semantic analysis
- Improved insight generation
- Documentation and examples

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

MIT License. See [LICENSE](LICENSE) for more information.

## 📫 Community & Support

- [Website](https://klaralabs.com)
- [GitHub Issues](https://github.com/yourusername/klarity/issues) for bugs and features
- [GitHub Discussions](https://github.com/yourusername/klarity/discussions) for questions