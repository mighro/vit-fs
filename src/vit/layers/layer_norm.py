import torch
from torch import Tensor, nn


class LayerNormalisation(nn.Module):
    """Normalize each token independently across its feature dimension.

    Args:
        embed_dim: Number of features in each token representation.
        eps: Small value added to the variance for numerical stability.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        """Initialize the layer normalization parameters.

        Args:
            embed_dim: Number of features in each token representation.
            eps: Small value added to the variance for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the input across its final dimension.

        Args:
            x: Input tensor whose final dimension has size `embed_dim`.

        Returns:
            A tensor with the same shape as `x`, normalized using the learnable
            scale (`alpha`) and bias parameters.
        """
        # Layer normalization computes the mean and variance across the final
        # dimension, then applies a learnable scale and bias. This is written
        # out explicitly instead of using torch.nn.LayerNorm so the operation
        # stays visible as part of the from-scratch implementation.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias
