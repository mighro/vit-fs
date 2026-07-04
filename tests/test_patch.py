import pytest
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


def test_patch_embedding_different_sizes():
    """Test with various image and patch sizes."""
    for img_size in [(32, 32), (64, 64), (128, 128)]:
        for p_size in [8, 16, 32]:
            if img_size[0] % p_size == 0 and img_size[1] % p_size == 0:
                pe = PatchEmbedding(
                    in_channels=3, patch_size=p_size, image_size=img_size, embed_dim=512
                )
                x = torch.randn(2, 3, *img_size)
                out = pe(x)
                expected_patches = (img_size[0] // p_size) * (img_size[1] // p_size)
                assert out.shape == (2, expected_patches, 512)


def test_patch_embedding_grayscale():
    """Test with single channel input."""
    pe = PatchEmbedding(
        in_channels=1, patch_size=16, image_size=(224, 224), embed_dim=768
    )
    x = torch.randn(1, 1, 224, 224)
    out = pe(x)
    assert out.shape == (1, 196, 768)


def test_patch_embedding_invalid_dimensions():
    """Ensure RuntimeError is raised when input dimensions aren't divisible by patch_size."""
    pe = PatchEmbedding(
        in_channels=3, patch_size=16, image_size=(224, 224), embed_dim=768
    )
    x = torch.randn(1, 3, 220, 220)  # Not divisible by 16
    with pytest.raises(RuntimeError, match="Input width/height must be divisible"):
        pe(x)


def test_patch_embedding_batch_size():
    """Test that batch dimension is correctly preserved."""
    pe = PatchEmbedding(
        in_channels=3, patch_size=16, image_size=(224, 224), embed_dim=768
    )
    for B in [1, 4, 8]:
        x = torch.randn(B, 3, 224, 224)
        out = pe(x)
        assert out.shape[0] == B
