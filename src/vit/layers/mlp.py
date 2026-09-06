from torch import Tensor, nn

EXPANSION_FACTOR = 8 / 3
MLP_DROPOUT = 0.1
MULTIPLE_OF = 256


class FeedForward(nn.Module):
    """Apply a SwiGLU feed-forward network to each token independently.

    The feed-forward network expands each token representation into a larger
    hidden dimension, applies a SiLU-based gating mechanism, and projects the
    result back to the original embedding dimension.

    Unlike the original ViT feed-forward network, which uses a conventional
    activation between two linear layers, this implementation uses the
    SwiGLU formulation:

        SiLU(W_gate(x)) * W_up(x)

    The hidden dimension is rounded up to a multiple of `multiple_of`.

    Args:
        embed_dim: Dimension of the input and output token representations.
        expansion_factor: Multiplier used to determine the hidden dimension
            before rounding.
        dropout_rate: Dropout probability applied after the output projection.
        multiple_of: Hidden dimension is rounded up to the nearest multiple
            of this value.
    """

    def __init__(
        self,
        embed_dim: int,
        expansion_factor: float = EXPANSION_FACTOR,
        dropout_rate: float = MLP_DROPOUT,
        multiple_of: int = MULTIPLE_OF,
    ):
        """Initialize the SwiGLU feed-forward network.

        Args:
            embed_dim: Dimension of the input and output token representations.
            expansion_factor: Multiplier used to calculate the intermediate
                hidden dimension.
            dropout_rate: Dropout probability applied after the output projection.
            multiple_of: Round the hidden dimension up to the nearest multiple
                of this value.
        """
        super().__init__()

        # Calculate hidden dimension and round up to the nearest multiple
        hidden_dim = int(embed_dim * expansion_factor)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=False)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated feed-forward transformation.

        Args:
            x: Input token representations with shape ``(B, N, D)``.

        Returns:
            Transformed token representations with the same shape as `x`.
        """
        hidden = self.act(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(hidden))
