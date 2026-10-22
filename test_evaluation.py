from types import SimpleNamespace

import pytest
import torch

from positivity_eval import evaluate_positivity
from src.steering.hooks import steer_residual, decode_sae_direction, extract_layer_features


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    def encode(self, text, **kwargs):
        return [0 if text.strip() == "A" else 1]

    def __call__(self, prompt, **kwargs):
        return Batch(input_ids=torch.tensor([[int("(A) positive" in prompt)]]))


class ChoiceModel:
    def __init__(self, positive_signal=0):
        self.positive_signal = positive_signal

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids):
        positive_first = bool(input_ids.item())
        scores = torch.tensor([4.0, 0.0])  # Fixed preference for option A.
        scores[0 if positive_first else 1] += self.positive_signal
        return SimpleNamespace(logits=scores.reshape(1, 1, 2))


@pytest.mark.parametrize("signal", [0.0, 2.0])
def test_option_position_bias_cancels(signal):
    dataset = [{"prompt": "Choose a description", "positive": "positive", "negative": "negative"}]
    result = evaluate_positivity(ChoiceModel(signal), Tokenizer(), dataset, device="cpu")
    assert result["avg_score"] == signal
    assert len(result["order_scores"]) == 1


def test_hook_preserves_output_structure_and_scales_per_example():
    hidden = torch.tensor([[[3.0, 4.0]], [[0.0, 10.0]]])
    direction = torch.tensor([1.0, 0.0])
    expected = torch.tensor([[[8.0, 4.0]], [[10.0, 10.0]]])
    torch.testing.assert_close(steer_residual(hidden, direction, 1.0), expected)
    cache = object()
    result = steer_residual((hidden, cache), direction, 1.0)
    torch.testing.assert_close(result[0], expected)
    assert result[1] is cache
    torch.testing.assert_close(hidden, torch.tensor([[[3.0, 4.0]], [[0.0, 10.0]]]))


def test_sae_vector_cannot_be_injected_without_decoding():
    with pytest.raises(ValueError, match="residual coordinates"):
        steer_residual(torch.ones(1, 3, 4), torch.ones(8), 1.0)


def test_latent_penalty_matches_overlap_after_decoding():
    latent = torch.tensor([1.0, 2.0, -1.0, 0.5], requires_grad=True)
    decoder = torch.arange(12.0).reshape(4, 3)
    unembedding = torch.tensor([1.0, -0.5, 0.25])
    decoded = decode_sae_direction(latent, decoder)
    assert decoded.shape == unembedding.shape
    torch.testing.assert_close(decoded @ unembedding, latent @ (decoder @ unembedding))
    (decoded @ unembedding).square().backward()
    assert latent.grad is not None


def test_features_use_post_layer_states_without_padding_or_autograd():
    states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])

    class FeatureTokenizer:
        def __call__(self, text, **kwargs):
            return Batch(input_ids=torch.zeros(1, 3, dtype=torch.long),
                         attention_mask=torch.tensor([[1, 1, 0]]))

    class FeatureModel:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            return SimpleNamespace(hidden_states=(torch.zeros_like(states), states))

    encoder = torch.nn.Linear(2, 4, bias=False)
    sae = SimpleNamespace(encode=encoder)
    raw = extract_layer_features(FeatureModel(), FeatureTokenizer(), "text", 0)
    torch.testing.assert_close(raw, torch.tensor([2.0, 3.0]))
    features = extract_layer_features(FeatureModel(), FeatureTokenizer(), "text", 0, sae=sae)
    assert features.shape == (4,)
    assert not features.requires_grad


def test_prototype_uses_the_causal_lm_output_head():
    # Extract this pure function without running the model-loading/training script.
    import ast
    from pathlib import Path
    source = ast.parse((Path(__file__).parent / "src/steering/c_perp.py").read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "get_positive_unembedding")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<unembedding-accessor>", "exec"), namespace)
    weights = torch.arange(12.0).reshape(4, 3)
    model = SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=weights))
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
    torch.testing.assert_close(namespace["get_positive_unembedding"](tokenizer, model, ["word"]), weights[1])
