# work in progress
import pytest
import numpy as np
from klarity.core.analyzer import EntropyAnalyzer
from klarity.models import TokenInfo, UncertaintyAnalysisRequest


@pytest.fixture
def analyzer():
    return EntropyAnalyzer(min_token_prob=0.01)


@pytest.fixture
def sample_token_info():
    return [
        TokenInfo(token="the", token_id=0, logit=0.0, probability=0.6),
        TokenInfo(token="a", token_id=1, logit=0.0, probability=0.3),
        TokenInfo(token="an", token_id=2, logit=0.0, probability=0.1),
    ]


@pytest.fixture
def sample_request(sample_token_info):
    return UncertaintyAnalysisRequest(
        logits=[0.0, 0.0, 0.0],
        prompt="test prompt",
        model_id="test_model",
        token_info=sample_token_info,
    )


def test_init():
    analyzer = EntropyAnalyzer(min_token_prob=0.01)
    assert analyzer.min_token_prob == 0.01
    assert analyzer.embedding_model is not None


def test_raw_entropy_calculation(analyzer):
    # Test with uniform distribution
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = analyzer._calculate_raw_entropy(uniform_probs)
    assert np.isclose(entropy, 1.0)  # Should be maximum entropy

    # Test with completely certain distribution
    certain_probs = np.array([1.0, 0.0, 0.0, 0.0])
    entropy = analyzer._calculate_raw_entropy(certain_probs)
    assert np.isclose(entropy, 0.0)  # Should be minimum entropy

    # Test with sample distribution
    sample_probs = np.array([0.6, 0.3, 0.1])
    entropy = analyzer._calculate_raw_entropy(sample_probs)
    assert 0.0 < entropy < 1.0  # Should be intermediate entropy


def test_semantic_entropy_calculation(analyzer, sample_token_info):
    # Test with similar tokens
    entropy = analyzer._calculate_semantic_entropy(sample_token_info)
    assert 0.0 <= entropy <= 1.0


def test_group_similar_tokens(analyzer, sample_token_info):
    # Create mock similarity matrix
    similarity_matrix = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0.3], [0.3, 0.3, 1.0]])

    groups = analyzer._group_similar_tokens(similarity_matrix, sample_token_info)
    assert len(groups) == 2  # Should group "the" and "a" together, "an" separate


def test_calculate_group_probabilities(analyzer, sample_token_info):
    groups = {
        0: [0, 1],  # "the" and "a"
        1: [2],  # "an"
    }

    probs = analyzer._calculate_group_probabilities(groups, sample_token_info)
    assert np.isclose(sum(probs.values()), 1.0)  # Probabilities should sum to 1
    assert len(probs) == 2  # Should have two groups


def test_analyze_empty_request(analyzer):
    empty_request = UncertaintyAnalysisRequest(
        logits=[], prompt="", model_id="test", token_info=[]
    )
    metrics = analyzer.analyze(empty_request)
    assert metrics.raw_entropy == 0.0
    assert metrics.semantic_entropy == 0.0


def test_analyze_single_token(analyzer):
    single_token = [TokenInfo(token="test", token_id=0, logit=0.0, probability=1.0)]
    request = UncertaintyAnalysisRequest(
        logits=[0.0], prompt="test", model_id="test", token_info=single_token
    )
    metrics = analyzer.analyze(request)
    assert metrics.raw_entropy == 0.0  # Single token should have zero entropy
    assert metrics.semantic_entropy == 0.0


def test_analyze_full_request(analyzer, sample_request):
    metrics = analyzer.analyze(sample_request)
    assert 0.0 <= metrics.raw_entropy <= 1.0
    assert 0.0 <= metrics.semantic_entropy <= 1.0
    assert len(metrics.token_predictions) == len(sample_request.token_info)
