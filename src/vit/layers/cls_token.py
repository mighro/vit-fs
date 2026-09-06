import torch
from torch import nn


def get_cls_token(embed_dim: int) -> nn.Parameter:
    """Create a learnable classification token.

    The classification token is prepended to the patch-token sequence and is
    trained to aggregate information from the entire image through
    self-attention. Its final representation is used by the classification
    head.

    Args:
        embed_dim: Dimension of the token embedding.

    Returns:
        A learnable parameter with shape ``(1, 1, embed_dim)``.
    """
    return nn.Parameter(
        torch.randn(1, 1, embed_dim),
        requires_grad=True,
    )
