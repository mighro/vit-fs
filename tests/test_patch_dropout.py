import pytest
import torch

from vit.layers import PatchDropout


def test_patch_dropout_eval_mode():
    """Test that PatchDropout acts as an identity layer during evaluation."""
    patch_dropout = PatchDropout(prob=0.5)
    patch_dropout.eval()
    x = torch.randn(4, 196, 768)
    out = patch_dropout(x)
    assert torch.allclose(out, x)


def test_patch_dropout_zero_probability():
    """Test that prob=0.0 returns the input unchanged during training."""
    patch_dropout = PatchDropout(prob=0.0)
    patch_dropout.train()
    x = torch.randn(4, 196, 768)
    out = patch_dropout(x)
    assert torch.allclose(out, x)


@pytest.mark.parametrize(
    "num_patches, prob",
    [(196, 0.5), (196, 0.25), (100, 0.1), (10, 0.9)],
)
def test_patch_dropout_output_shape(num_patches, prob):
    """Test that training keeps max(1, int(N * (1 - prob))) patch tokens."""
    batch_size = 2
    embed_dim = 32

    patch_dropout = PatchDropout(prob=prob)
    patch_dropout.train()
    x = torch.randn(batch_size, num_patches, embed_dim)
    out = patch_dropout(x)

    expected_keep = max(1, int(num_patches * (1.0 - prob)))
    assert out.shape == (batch_size, expected_keep, embed_dim)


def test_patch_dropout_keeps_whole_tokens():
    """Test that entire token vectors are kept unchanged, not individual features."""
    torch.manual_seed(42)
    num_patches = 16
    embed_dim = 8

    patch_dropout = PatchDropout(prob=0.5)
    patch_dropout.train()

    # Give every patch token a distinct, constant value so it can be identified.
    x = torch.arange(num_patches, dtype=torch.float32)
    x = x.view(1, num_patches, 1).expand(1, num_patches, embed_dim).clone()

    out = patch_dropout(x)

    # Every kept token should still be constant across the embedding dimension.
    for token in out[0]:
        assert torch.all(token == token[0])

    # Kept tokens should be a subset of the original tokens with no duplicates.
    kept_values = out[0, :, 0].tolist()
    assert set(kept_values).issubset(set(range(num_patches)))
    assert len(set(kept_values)) == out.shape[1]


def test_patch_dropout_minimum_one_token():
    """Test that patch dropout never drops the entire sequence."""
    patch_dropout = PatchDropout(prob=0.99)
    patch_dropout.train()
    x = torch.randn(2, 8, 16)
    out = patch_dropout(x)
    assert out.shape == (2, 1, 16)


def test_patch_dropout_selection_varies():
    """Test that different forward passes drop different tokens during training."""
    torch.manual_seed(0)
    num_patches = 196

    patch_dropout = PatchDropout(prob=0.5)
    patch_dropout.train()
    x = torch.randn(1, num_patches, 16)

    out_1 = patch_dropout(x)
    out_2 = patch_dropout(x)

    assert not torch.equal(out_1, out_2)


def test_patch_dropout_gradient_flow():
    """Test that gradients flow only through the kept patch tokens."""
    torch.manual_seed(0)

    patch_dropout = PatchDropout(prob=0.5)
    patch_dropout.train()
    x = torch.randn(1, 12, 4, requires_grad=True)

    out = patch_dropout(x)
    out.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Kept tokens receive a gradient of 1, dropped tokens receive 0.
    assert set(x.grad.unique().tolist()) == {0.0, 1.0}
    assert x.grad.sum() == out.shape[1] * out.shape[2]


@pytest.mark.parametrize("prob", [-0.1, 1.0, 1.5])
def test_patch_dropout_invalid_probability(prob):
    """Ensure probabilities outside [0.0, 1.0) raise ValueError."""
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        PatchDropout(prob=prob)
