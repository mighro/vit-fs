from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    """
    Configuration parameters for the Vision Transformer (ViT).

    This dataclass defines the hyperparameters, architectural dimensions,
    and regularization settings used to instantiate a ViT model. It also
    provides class methods to quickly initialize standard model scales
    (tiny, base, large, xlarge).

    Attributes:
        patch_size (int): The height and width of each image patch (e.g., 16
            creates 16x16 pixel patches). Default: 16.
        embed_dim (int): The dimensionality of the token embeddings and the
            hidden state throughout the transformer layers. Default: 768.
        head_size (int): The dimensionality of each attention head. The total
            number of heads is implicitly `embed_dim // head_size`. Default: 64.
        depth (int): The number of Transformer encoder blocks (layers) in the
            network. Default: 12.
        mlp_ratio (float): The multiplier used to determine the hidden dimension
            of the MLP/FeedForward block relative to the `embed_dim`
            (i.e., mlp_hidden_dim = embed_dim * mlp_ratio). Default: 8/3.
        mlp_drop (float): The dropout probability applied to the activations
            within the MLP blocks. Default: 0.1.
        patch_drop (float): The dropout probability applied immediately after
            the initial patch embedding. Default: 0.1.
        attn_proj_drop (float): The dropout probability applied to the final
            output projection of the Multi-Head Attention module. Default: 0.1.
        attn_drop (float | None): The dropout probability applied directly to
            the computed attention matrix (attention weights). Default: None.
        max_path_drop (float): The maximum probability for Stochastic Depth
            (DropPath). The drop rate linearly increases from 0.0 at the first
            layer to `max_path_drop` at the final layer. Default: 0.15.
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
