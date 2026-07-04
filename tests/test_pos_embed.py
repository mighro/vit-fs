import torch
from vit.layers.pos_embed import PosEmbedding


def test_pos_embedding():
    pos_embedding = PosEmbedding(num_positions=196, embed_dim=768)
    x = torch.randn(1, 196, 768)
    output = pos_embedding(x)
    assert output.shape == (1, 196, 768)


def test_pos_embedding_different_sizes():
    """Test with various sequence lengths and embedding dimensions."""
    for seq_len in [10, 50, 256]:
        for embed_dim in [64, 256, 512]:
            pe = PosEmbedding(num_positions=seq_len, embed_dim=embed_dim)
            x = torch.randn(2, seq_len, embed_dim)
            output = pe(x)
            assert output.shape == (2, seq_len, embed_dim)


def test_pos_embedding_gradient_flow():
    """Ensure positional embeddings are trainable."""
    pe = PosEmbedding(num_positions=10, embed_dim=32)
    x = torch.randn(1, 10, 32)
    output = pe(x)
    loss = output.sum()
    loss.backward()
    assert pe.pos_embeddings.grad is not None


def test_pos_embedding_addition_correctness():
    """Verify that output is exactly input + positional embedding."""
    pe = PosEmbedding(num_positions=5, embed_dim=4)
    x = torch.ones(1, 5, 4)
    output = pe(x)
    assert torch.allclose(output - x, pe.pos_embeddings)


def test_pos_embedding_batch_independence():
    """Test that positional embeddings are applied identically across batch items."""
    pe = PosEmbedding(num_positions=3, embed_dim=2)
    x = torch.randn(4, 3, 2)
    output = pe(x)
    # The difference between output and input should be the same for all batch items
    diff = output - x
    assert torch.allclose(diff[0], diff[1])
    assert torch.allclose(diff[1], diff[2])
    assert torch.allclose(diff[2], diff[3])
