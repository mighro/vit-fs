from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    """Configuration for a Vision Transformer.

    This immutable configuration object contains the architectural and
    regularization hyperparameters used to construct a `ClassificationViT`.

    The configuration controls the patch size, token embedding dimension,
    number of transformer blocks, attention-head size, feed-forward expansion,
    dropout rates, and stochastic-depth schedule.

    The `tiny()`, `base()`, `large()`, and `xlarge()` class methods provide
    convenient predefined configurations. These are project-specific presets
    rather than exact reproductions of the configurations from the original
    ViT paper.

    Attributes:
        patch_size: Height and width of each image patch.
        embed_dim: Dimension of every token representation throughout the
            transformer.
        head_size: Dimension of each attention head. The number of heads is
            computed as `embed_dim // head_size`.
        depth: Number of transformer encoder blocks.
        mlp_ratio: Multiplier used to determine the hidden dimension of the
            feed-forward network.
        mlp_drop: Dropout probability applied by the feed-forward network.
        patch_drop: Dropout probability applied after patch embedding.
        attn_proj_drop: Dropout probability applied after the attention
            output projection.
        attn_drop: Dropout probability applied to the attention weights.
            If `None`, the attention module chooses a rate automatically.
        max_path_drop: Maximum stochastic-depth probability. The probability
            increases linearly from the first to the final transformer block.
    """

    # Patch embedding
    patch_size: int = 16

    # Architecture dimensions
    embed_dim: int = 768
    head_size: int = 64
    depth: int = 12

    # Ratios and multiples
    mlp_ratio: float = 8 / 3

    # Dropout settings
    mlp_drop: float = 0.1
    patch_drop: float = 0.1
    attn_proj_drop: float = 0.1
    attn_drop: float | None = None
    max_path_drop: float = 0.15

    @classmethod
    def tiny(cls, **kwargs) -> "ViTConfig":
        return cls(embed_dim=256, depth=10, max_path_drop=0.05, **kwargs)

    @classmethod
    def base(cls, **kwargs) -> "ViTConfig":
        return cls(embed_dim=384, depth=12, max_path_drop=0.1, **kwargs)

    @classmethod
    def large(cls, **kwargs) -> "ViTConfig":
        return cls(embed_dim=512, depth=14, max_path_drop=0.15, **kwargs)

    @classmethod
    def xlarge(cls, **kwargs) -> "ViTConfig":
        return cls(embed_dim=768, depth=16, max_path_drop=0.2, **kwargs)
