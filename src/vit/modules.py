import torch
from torch import Tensor, nn

from .layers import (
    FeedForward,
    LayerNormalisation,
    MultiHeadSelfAttention,
    PatchEmbedding,
    PosEmbedding,
    get_cls_token,
)


class ViTEmbeddings(nn.Module):
    """ViT input layer combining patch embedding, position encoding, and CLS token."""

    def __init__(
        self,
        in_channels: int,
        image_size: tuple[int, int],
        patch_size: int,
        embed_dim: int,
        patch_dropout_rate: float,
    ):
        """Initialize ViT input layer.

        Args:
            in_channels: Number of input image channels (e.g., 3 for RGB).
            image_size: Input image dimensions as (height, width).
            patch_size: Size of each square patch.
            embed_dim: Dimension of patch embeddings.
            patch_dropout_rate: Dropout probability for patch embeddings.
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            image_size=image_size,
            embed_dim=embed_dim,
        )
        self.pos_embed = PosEmbedding(
            num_positions=self.patch_embed.num_patches,
            embed_dim=embed_dim,
        )
        self.cls_token = get_cls_token(embed_dim=embed_dim)
        self.patch_dropout = nn.Dropout(patch_dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        # Ensure batch dimension exists. Expects (C, H, W) or (B, C, H, W).
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = self.patch_embed(x)
        x = self.patch_dropout(x)
        x = self.pos_embed(x)

        # Expand CLS token to match batch size before concatenation
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat((cls_tokens, x), dim=1)


class EncoderBlock(nn.Module):
    """Standard Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        head_size: int,
        mlp_ratio: float,
        mlp_drop: float,
        proj_drop: float,
        attn_drop: float | None = None,
    ):
        """Initialize the encoder block.

        Args:
            embed_dim: Dimension of the token embeddings.
            head_size: Hidden dimension size per individual attention head.
            mlp_ratio: Expansion factor for the feed-forward network's hidden layer.
            mlp_drop: Dropout probability applied within the feed-forward network.
            proj_drop: Dropout probability applied to the attention projection output.
            attn_drop: Dropout probability applied to attention weights (None for auto).
        """
        super().__init__()

        self.attn_norm = LayerNormalisation(embed_dim)
        self.mlp_norm = LayerNormalisation(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, head_size, proj_drop, attn_drop)
        self.mlp = FeedForward(embed_dim, mlp_ratio, mlp_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))

        return x
