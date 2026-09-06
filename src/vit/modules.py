import torch
from torch import Tensor, nn

from .layers import (
    DropPath,
    FeedForward,
    LayerNormalisation,
    MultiHeadSelfAttention,
    PatchEmbedding,
    PosEmbedding,
    get_cls_token,
)


class ViTEmbeddings(nn.Module):
    """Convert an image into the token sequence consumed by the Transformer.

    This module performs the input stage of the Vision Transformer:

    1. Convert the image into patch embeddings.
    2. Apply patch dropout.
    3. Add learnable positional embeddings.
    4. Prepend a learnable `[CLS]` token.

    The resulting sequence is passed directly to the Transformer encoder
    blocks.

    Args:
        in_channels: Number of channels in the input image.
        image_size: Expected input image dimensions as ``(height, width)``.
        patch_size: Height and width of each square image patch.
        embed_dim: Dimension of every token embedding.
        patch_dropout_rate: Dropout probability applied to patch embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: tuple[int, int],
        patch_size: int,
        embed_dim: int,
        patch_dropout_rate: float,
    ):
        """Initialize the ViT input embedding pipeline.

        Args:
            in_channels: Number of channels in the input image.
            image_size: Expected input image dimensions as ``(height, width)``.
            patch_size: Height and width of each square image patch.
            embed_dim: Dimension of the patch and token embeddings.
            patch_dropout_rate: Dropout probability applied after patch
                embedding.
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
        """Convert an image into a Transformer-ready token sequence.

        A missing batch dimension is automatically added when `x` has shape
        ``(C, H, W)``.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)`` or ``(C, H, W)``.

        Returns:
            Token sequence with shape ``(B, N + 1, D)``, where `N` is the number
            of image patches and the additional token is the `[CLS]` token.
        """
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
    """Transformer encoder block using pre-normalization and residual paths.

    The block contains two sublayers:

    1. Multi-head self-attention.
    2. SwiGLU feed-forward network.

    Each sublayer is preceded by layer normalization and followed by a
    residual connection. DropPath can optionally regularize both residual
    branches using stochastic depth.

    The resulting structure is:

        x = x + DropPath(Attention(Norm(x)))
        x = x + DropPath(FeedForward(Norm(x)))

    Args:
        embed_dim: Dimension of the token representations.
        head_size: Dimension of each attention head.
        mlp_ratio: Expansion factor used by the feed-forward network.
        mlp_drop: Dropout probability used by the feed-forward network.
        proj_drop: Dropout probability applied after attention's output
            projection.
        drop_path: Stochastic-depth probability for the residual branches.
        attn_drop: Dropout probability applied to attention weights. If
            `None`, the attention module determines the rate automatically.
    """

    def __init__(
        self,
        embed_dim: int,
        head_size: int,
        mlp_ratio: float,
        mlp_drop: float,
        proj_drop: float,
        drop_path: float,
        attn_drop: float | None = None,
    ):
        """Initialize a Transformer encoder block.

        Args:
            embed_dim: Dimension of the token representations.
            head_size: Dimension of each attention head.
            mlp_ratio: Expansion factor used by the SwiGLU feed-forward network.
            mlp_drop: Dropout probability used by the feed-forward network.
            proj_drop: Dropout probability applied after attention's output
                projection.
            drop_path: Stochastic-depth probability applied to each residual
                branch.
            attn_drop: Dropout probability applied to attention weights.
                If `None`, the attention module determines the rate automatically.
        """
        super().__init__()

        self.attn_norm = LayerNormalisation(embed_dim)
        self.mlp_norm = LayerNormalisation(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, head_size, proj_drop, attn_drop)
        self.mlp = FeedForward(embed_dim, mlp_ratio, mlp_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention and feed-forward transformations to the tokens.

        Args:
            x: Input token representations with shape ``(B, N, D)``.

        Returns:
            Transformed token representations with the same shape as `x`.
        """
        x = x + self.drop_path(self.attn(self.attn_norm(x)))
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        return x
