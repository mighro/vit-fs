import torch
from torch import Tensor, nn

HEAD_SIZE = 64
PROJ_DROP = 0.1


class MultiHeadSelfAttention(nn.Module):
    """Computes Multi-Head Self-Attention on an input sequence."""

    def __init__(
        self,
        embed_dim: int,
        head_size: int = HEAD_SIZE,
        proj_drop: float = PROJ_DROP,
        attn_drop: float | None = None,
    ):
        """Initializes the MultiHeadSelfAttention module.

        Args:
            embed_dim: Feature dimension of each token vector.
            head_size: Hidden dimension size per individual attention head.
            proj_drop: Dropout probability applied to the final output.
            attn_drop: Dropout probability applied to attention weights.
                If None, calculates rate dynamically based on head count.

        Raises:
            ValueError: If embed_dim is not exactly divisible by head_size.
        """
        super().__init__()

        if embed_dim % head_size != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be exactly divisible "
                f"by head_size ({head_size})."
            )

        self.n_heads = embed_dim // head_size
        self.d_k = head_size

        self.scale = self.d_k**-0.5

        self.w_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim)

        if attn_drop is None:
            attn_drop = 0.1 * (1 + 0.05 * (self.n_heads - 8))
            attn_drop = max(0.05, min(0.2, attn_drop))

        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.proj_dropout = nn.Dropout(p=proj_drop)

    @staticmethod
    def attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scale: float,
        dropout: nn.Dropout | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Scaled dot-product attention over Queries, Keys, and Values."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.softmax(dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        return torch.matmul(scores, v), scores

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape

        q, k, v = (
            self.w_qkv(x)
            .reshape(B, N, 3, self.n_heads, self.d_k)
            .permute(
                2, 0, 3, 1, 4
            )  # (B, N, 3, num_heads, d_k) -> (3, B, num_heads, N, d_k)
        )

        x, self.attention_score = self.attention(q, k, v, self.scale, self.attn_dropout)

        x = x.transpose(1, 2).contiguous().view(B, N, self.n_heads * self.d_k)

        return self.proj_dropout(self.w_o(x))
