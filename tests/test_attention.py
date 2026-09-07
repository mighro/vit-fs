import pytest
import torch

from vit.layers import MultiHeadSelfAttention


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


def test_multi_head_self_attention_invalid_backend():
    """Ensure ValueError is raised when an unsupported backend is requested."""
    with pytest.raises(ValueError, match="Invalid attention backend"):
        MultiHeadSelfAttention(embed_dim=768, head_size=64, backend="unsupported")


def test_multi_head_self_attention_dropout_behavior():
    """Test that full attention dropout produces a near-zero output."""
    embed_dim = 768
    head_size = 64
    attn = MultiHeadSelfAttention(
        embed_dim=embed_dim,
        head_size=head_size,
        attn_drop=1.0,
        proj_drop=1.0,
        backend="manual",
    )
    x = torch.randn(1, 10, embed_dim)
    out = attn(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_attention_score_storage():
    """Verify that attention scores are stored in manual mode."""
    embed_dim = 768
    head_size = 64
    attn = MultiHeadSelfAttention(
        embed_dim=embed_dim, head_size=head_size, backend="manual"
    )
    B, N = 1, 10
    x = torch.randn(B, N, embed_dim)
    _ = attn(x)
    assert hasattr(attn, "attention_score")
    assert attn.attention_score is not None
    assert attn.attention_score.shape == (B, 12, N, N)


def test_attention_backends_consistency():
    """Verify both 'torch' and 'manual' backends produce expected output shapes."""
    embed_dim = 256
    head_size = 64
    x = torch.randn(2, 8, embed_dim)

    attn_torch = MultiHeadSelfAttention(
        embed_dim=embed_dim, head_size=head_size, backend="torch"
    )
    attn_manual = MultiHeadSelfAttention(
        embed_dim=embed_dim, head_size=head_size, backend="manual"
    )

    out_torch = attn_torch(x)
    out_manual = attn_manual(x)

    assert out_torch.shape == x.shape
    assert out_manual.shape == x.shape
    assert attn_torch.attention_score is None
    assert attn_manual.attention_score is not None
