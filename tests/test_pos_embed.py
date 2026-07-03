import torch

from vit.layers.pos_embed import PosEmbedding


def test_pos_embedding():
    pos_embedding = PosEmbedding(num_positions=196, embed_dim=768)
    x = torch.randn(1, 196, 768)
    output = pos_embedding(x)
    assert output.shape == (1, 196, 768)
