import torch

from vit.modules import ViTEmbeddings


def test_vit_embeddings_shape():
    """Test that CLS token is correctly prepended and sequence length increases by 1."""
    embed_dim = 768
    patch_size = 16
    image_size = (224, 224)

    embeddings = ViTEmbeddings(
        in_channels=3,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        patch_dropout_rate=0.0,
    )

    B = 4
    x = torch.randn(B, 3, *image_size)
    out = embeddings(x)

    num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    assert out.shape == (B, num_patches + 1, embed_dim)


def test_vit_embeddings_3d_input_handling():
    """Test that 3D input (C, H, W) is automatically expanded to 4D."""
    embed_dim = 768
    patch_size = 16
    image_size = (224, 224)

    embeddings = ViTEmbeddings(
        in_channels=3,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        patch_dropout_rate=0.0,
    )

    x_3d = torch.randn(3, *image_size)  # Missing batch dimension
    out = embeddings(x_3d)

    assert out.shape[0] == 1  # Batch dim should be added


def test_vit_embeddings_cls_token_batch_expansion():
    """Verify CLS token is correctly expanded to match batch size."""
    embed_dim = 768
    patch_size = 16
    image_size = (224, 224)

    embeddings = ViTEmbeddings(
        in_channels=3,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        patch_dropout_rate=0.0,
    )

    B = 8
    x = torch.randn(B, 3, *image_size)
    out = embeddings(x)

    cls_tokens = out[:, 0, :]
    assert cls_tokens.shape == (B, embed_dim)


def test_vit_embeddings_num_patches_consistency():
    """Ensure num_patches matches expected calculation."""
    embed_dim = 768
    patch_size = 16
    image_size = (224, 224)

    embeddings = ViTEmbeddings(
        in_channels=3,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        patch_dropout_rate=0.0,
    )

    expected_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    assert embeddings.patch_embed.num_patches == expected_patches
