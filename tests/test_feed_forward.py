import pytest
import torch
from src.vit.layers.mlp import FeedForward


def test_feed_forward_output_shape():
    """Test that the output shape matches the input shape."""
    batch_size = 4
    seq_len = 12
    embed_dim = 768
    
    model = FeedForward(embed_dim=embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    out = model(x)
    assert out.shape == (batch_size, seq_len, embed_dim)


def test_feed_forward_hidden_dim_rounding():
    """Test that hidden dimension is correctly rounded up to the nearest multiple_of."""
    embed_dim = 100
    multiple_of = 64
    
    model = FeedForward(embed_dim=embed_dim, multiple_of=multiple_of)
    
    # 100 * (8/3) = 266.66 -> int -> 266
    # ceil(266 / 64) * 64 = 5 * 64 = 320
    expected_hidden = 320
    
    assert model.w_gate.out_features == expected_hidden
    assert model.w_up.out_features == expected_hidden
    assert model.w_down.in_features == expected_hidden


def test_feed_forward_custom_expansion_factor():
    """Test that custom expansion factors correctly scale the hidden dimension."""
    embed_dim = 512
    expansion_factor = 2.0
    
    model = FeedForward(embed_dim=embed_dim, expansion_factor=expansion_factor)
    
    # 512 * 2.0 = 1024 -> rounded to 1024 (multiple of 256)
    assert model.w_gate.out_features == 1024


def test_feed_forward_no_bias():
    """Test that all linear layers are initialized without bias."""
    embed_dim = 256
    model = FeedForward(embed_dim=embed_dim)
    
    assert model.w_gate.bias is None
    assert model.w_up.bias is None
    assert model.w_down.bias is None
