# work in progess
import pytest
import torch

from klarity.core.analyzer import EntropyAnalyzer
from klarity.estimator import UncertaintyEstimator, UncertaintyLogitsProcessor


class MockTokenizer:
    def decode(self, token_id, **kwargs):
        return f"token_{token_id}"


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def estimator():
    return UncertaintyEstimator(top_k=5, analyzer=EntropyAnalyzer(min_token_prob=0.02))


@pytest.fixture
def logits_processor(estimator):
    return estimator.get_logits_processor()


@pytest.fixture
def sample_logits():
    # Create sample logits tensor with shape [batch_size=1, vocab_size=5]
    return torch.tensor([[1.0, 2.0, 0.5, 3.0, 1.5]]).float()


@pytest.fixture
def sample_generation_output():
    class MockGenerationOutput:
        def __init__(self):
            self.sequences = torch.tensor([[1, 2, 3]])

    return MockGenerationOutput()


def test_init():
    estimator = UncertaintyEstimator(top_k=100, analyzer=EntropyAnalyzer(min_token_prob=0.02))
    assert estimator.top_k == 100, "top_k should be 100"
    assert isinstance(estimator.analyzer, EntropyAnalyzer), "analyzer should be an instance of EntropyAnalyzer"

    custom_analyzer = EntropyAnalyzer(min_token_prob=0.02)
    estimator = UncertaintyEstimator(top_k=50, analyzer=custom_analyzer)
    assert estimator.top_k == 50
    assert estimator.analyzer == custom_analyzer


def test_logits_processor_init(estimator):
    processor = estimator.get_logits_processor()
    assert isinstance(processor, UncertaintyLogitsProcessor)
    assert len(processor.captured_logits) == 0


def test_logits_processor_call(logits_processor, sample_logits):
    input_ids = torch.tensor([[1, 2, 3]])
    output = logits_processor(input_ids, sample_logits)

    assert torch.equal(output, sample_logits)
    assert len(logits_processor.captured_logits) == 1
    assert torch.equal(logits_processor.captured_logits[0], sample_logits)


def test_process_logits(
    estimator: UncertaintyEstimator,
    mock_tokenizer: MockTokenizer,
    sample_logits: torch.Tensor,
):
    token_info = estimator._process_logits(sample_logits, mock_tokenizer)

    assert len(token_info) == estimator.top_k
    # Check probabilities sum approximately to 1
    total_prob = sum(t.probability for t in token_info)
    assert 0.99 <= total_prob <= 1.01


def test_analyze_generation(estimator, mock_tokenizer, sample_generation_output, logits_processor):
    # Add some sample logits
    logits_processor.captured_logits.append(torch.randn(1, 5))
    logits_processor.captured_logits.append(torch.randn(1, 5))

    metrics_list = estimator.analyze_generation(
        sample_generation_output, tokenizer=mock_tokenizer, processor=logits_processor
    ).token_metrics

    assert len(metrics_list) == len(logits_processor.captured_logits)
    for metrics in metrics_list:
        assert 0.0 <= metrics.raw_entropy <= 1.0
        assert 0.0 <= metrics.semantic_entropy <= 1.0
        assert len(metrics.token_predictions) == estimator.top_k


def test_empty_generation(estimator, mock_tokenizer, sample_generation_output, logits_processor):
    metrics_list = estimator.analyze_generation(
        sample_generation_output, tokenizer=mock_tokenizer, processor=logits_processor
    ).token_metrics
    assert len(metrics_list) == 0


def test_process_logits_top_k(estimator, mock_tokenizer):
    # Test that we only get top_k predictions
    large_logits = torch.randn(1, 1000)  # Larger than top_k
    token_info = estimator._process_logits(large_logits, mock_tokenizer)
    assert len(token_info) == estimator.top_k


def test_process_logits_probabilities(estimator, mock_tokenizer, sample_logits):
    token_info = estimator._process_logits(sample_logits, mock_tokenizer)

    # Check probabilities are sorted
    probs = [t.probability for t in token_info]
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    # Check probabilities are valid
    assert all(0 <= p <= 1 for p in probs)
