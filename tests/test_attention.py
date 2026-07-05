import pytest
import torch

from vit.layers.attention import MultiHeadSelfAttention


def test_multi_head_self_attention_shape():
    """Test that output shape matches input shape."""
    embed_dim = 768
    head_size = 64
    attn = MultiHeadSelfAttention(embed_dim=embed_dim, head_size=head_size)

    B, N = 2, 196
    x = torch.randn(B, N, embed_dim)
    out = attn(x)

    assert out.shape == (B, N, embed_dim)


def test_multi_head_self_attention_heads_and_dk():
    """Verify correct number of heads and key dimension."""
    embed_dim = 768
    head_size = 64
    attn = MultiHeadSelfAttention(embed_dim=embed_dim, head_size=head_size)

    assert attn.n_heads == 12
    assert attn.d_k == 64


def test_multi_head_self_attention_invalid_dims():
    """Ensure ValueError is raised when embed_dim isn't divisible by head_size."""
    with pytest.raises(ValueError, match="embed_dim.*must be exactly divisible"):
        MultiHeadSelfAttention(embed_dim=700, head_size=64)


def test_multi_head_self_attention_dropout_behavior():
    """Test that 100% attention dropout results in near-zero output (before projection)."""
    embed_dim = 768
    head_size = 64
    # Set both dropouts to 1.0 to isolate the effect
    attn = MultiHeadSelfAttention(
        embed_dim=embed_dim,
        head_size=head_size,
        attn_drop=1.0,
        proj_drop=1.0,
    )

    x = torch.randn(1, 10, embed_dim)
    out = attn(x)

    # With full dropout on attention weights and projection, output should be zeros
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_attention_score_storage():
    """Verify that attention scores are correctly stored and shaped."""
    embed_dim = 768
    head_size = 64
    attn = MultiHeadSelfAttention(embed_dim=embed_dim, head_size=head_size)

    B, N = 1, 10
    x = torch.randn(B, N, embed_dim)
    _ = attn(x)

    assert hasattr(attn, "attention_score")
    # Attention scores shape: (B, num_heads, seq_len, seq_len)
    assert attn.attention_score.shape == (B, 12, N, N)
