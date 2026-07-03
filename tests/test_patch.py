import torch

from vit.layers.patch_embed import PatchEmbedding


def test_patch_embedding():
    patch_embedding = PatchEmbedding(
        in_channels=3, patch_size=16, image_size=(224, 224), embed_dim=768
    )
    x = torch.randn(1, 3, 224, 224)
    output = patch_embedding(x)
    assert output.shape == (1, 196, 768)
    assert patch_embedding.num_patches == 196
