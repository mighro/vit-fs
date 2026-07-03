import torch
from torch import Tensor, nn


class LayerNormalisation(nn.Module):
    """Applies LayerNorm independently to each token vector."""

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        """Initializes the LayerNormalisation module.

        Args:
            embed_dim: Feature dimension of each token vector.
            eps: Small constant for numerical stability in division.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        # Use variance with unbiased=False to match standard LayerNorm (ddof=0)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias
