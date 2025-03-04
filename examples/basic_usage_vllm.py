import json
import os

from dotenv import load_dotenv
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from klarity import UncertaintyEstimator
from klarity.core.analyzer import EntropyAnalyzer

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
llm = LLM(model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

estimator = UncertaintyEstimator(
    top_k=5,
    analyzer=EntropyAnalyzer(
        insight_model=llm,
        insight_tokenizer=tokenizer,
        min_token_prob=0.01,
    ),
)

prompt = "What is the capital of France?"
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=20, logprobs=5, prompt_logprobs=5)

outputs = llm.generate([prompt], sampling_params)
output = outputs[0]  # Get first output since we only sent one prompt

# Access the completion output which contains the generated text and logprobs
completion = output.outputs[0]
generated_text = completion.text

result = estimator.analyze_generation(output, tokenizer=tokenizer, prompt=prompt)

print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}")

for idx, metrics in enumerate(result.token_metrics):
    print(f"\nStep {idx}:")
    print(f"Raw entropy: {metrics.raw_entropy:.4f}")
    print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
    for pred in metrics.token_predictions[:3]:
        print(f"  {pred.token} (prob: {pred.probability:.4f})")

if result.overall_insight:
    print("\nAnalysis:")
    response = json.loads(result.overall_insight)
    print(json.dumps(response, indent=2))
