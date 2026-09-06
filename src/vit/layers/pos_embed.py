import torch
from torch import Tensor, nn


class PosEmbedding(nn.Module):
    """Add learnable positional embeddings to a sequence of tokens.

    Transformers do not inherently know the spatial or sequential position
    of a token. This layer provides that information by learning one embedding
    vector for each token position.

    The same positional embeddings are added to every item in the batch.

    Args:
        num_positions: Number of token positions represented by the embedding.
        embed_dim: Dimension of each token embedding.
    """

    def __init__(
        self,
        num_positions: int,
        embed_dim: int,
    ) -> None:
        """Initialize the learnable positional embedding table.

        Args:
            num_positions: Number of token positions represented by the embedding.
            embed_dim: Dimension of each token embedding.
        """
        super().__init__()

        self.pos_embeddings = nn.Parameter(
            torch.randn(1, num_positions, embed_dim),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embeddings to the input token sequence.

        Args:
            x: Token embeddings with shape ``(B, N, D)``, where ``B`` is the
                batch size, ``N`` is the number of positions, and ``D`` is
                `embed_dim`.

        Returns:
            Token embeddings with the same shape as `x`, after adding the
            corresponding positional embedding to each position.
        """
        return x + self.pos_embeddings
