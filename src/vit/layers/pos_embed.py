import torch
from torch import Tensor, nn


class PosEmbedding(nn.Module):
    """Pos embedding layer for Vision Transformers."""

    def __init__(
        self,
        num_positions: int,
        embed_dim: int,
    ) -> None:
        """Pos embedding layer for Vision Transformers.

        Args:
            num_positions: Number of positions to embed.
            embed_dim: Dimension of the embedding.
        """
        super().__init__()

        self.pos_embeddings = nn.Parameter(
            torch.randn(1, num_positions, embed_dim),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_embeddings
