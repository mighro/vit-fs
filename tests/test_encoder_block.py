import torch

from vit.modules import EncoderBlock


def test_encoder_block_output_shape():
    batch_size = 2
    seq_len = 10
    embed_dim = 768
    head_size = 64
    mlp_ratio = 4.0

    block = EncoderBlock(
        embed_dim=embed_dim,
        head_size=head_size,
        mlp_ratio=mlp_ratio,
        mlp_drop=0.0,
        drop_path=0.0,
        proj_drop=0.0,
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    out = block(x)

    assert out.shape == x.shape
