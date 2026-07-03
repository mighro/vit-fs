import torch
from torch import Tensor, nn


def get_cls_token(embed_dim: int) -> nn.Parameter:
    """Returns a learnable CLS token parameter of shape (1, 1, embed_dim)."""
    return nn.Parameter(
        torch.randn(1, 1, embed_dim),
        requires_grad=True,
    )
