from torch import Tensor, nn

EXPANSION_FACTOR = 8 / 3
MLP_DROPOUT = 0.1
MULTIPLE_OF = 256


class FeedForward(nn.Module):
    """Applies a Swish-Gated Linear Unit (SwiGLU) feed-forward network."""

    def __init__(
        self,
        embed_dim: int,
        expansion_factor: float = EXPANSION_FACTOR,
        dropout_rate: float = MLP_DROPOUT,
        multiple_of: int = MULTIPLE_OF,
    ):
        """Initializes the SwiGLU FeedForward module.

        Args:
            embed_dim: The feature dimension of the input and output token vectors.
            expansion_factor: Multiplier determining the internal hidden dimension size.
            dropout_rate: Dropout probability applied to the final output projection.
            multiple_of: Rounds the hidden dimension up to a multiple of this value
                for optimal GPU memory alignment.
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
        hidden = self.act(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(hidden))
