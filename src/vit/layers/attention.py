import torch
from torch import Tensor, nn

HEAD_SIZE = 64
PROJ_DROP = 0.1


class MultiHeadSelfAttention(nn.Module):
    """Compute multi-head self-attention over a sequence of tokens.

    Each input token is projected into a Query, Key, and Value representation.
    The queries and keys are used to calculate scaled attention scores, which
    determine how strongly each token attends to every other token.

    The embedding dimension is divided across multiple attention heads.
    Each head performs attention independently before the results are
    concatenated and projected back to the original embedding dimension.

    The computed attention weights are stored in `attention_score` after each
    forward pass, which can be useful for visualization and analysis.

    Args:
        embed_dim: Dimension of each input and output token representation.
        head_size: Dimension of each individual attention head.
        proj_drop: Dropout probability applied after the output projection.
        attn_drop: Dropout probability applied to the attention weights.
            If `None`, a rate is selected based on the number of heads.

    Raises:
        ValueError: If `embed_dim` is not evenly divisible by `head_size`.
    """

    def __init__(
        self,
        embed_dim: int,
        head_size: int = HEAD_SIZE,
        proj_drop: float = PROJ_DROP,
        attn_drop: float | None = None,
    ):
        """Initialize the multi-head self-attention module.

        Args:
            embed_dim: Dimension of each input and output token representation.
            head_size: Dimension of each attention head.
            proj_drop: Dropout probability applied after the output projection.
            attn_drop: Dropout probability applied to the attention weights.
                If `None`, the dropout rate is determined automatically.

        Raises:
            ValueError: If `embed_dim` is not evenly divisible by `head_size`.
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
        """Compute scaled dot-product attention.

        Attention scores are calculated from the similarity between Queries and
        Keys, scaled by ``1 / sqrt(d_k)``, and normalized with softmax. The
        resulting attention weights are then used to compute a weighted sum of
        the Values.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            scale: Scaling factor applied to the Query-Key dot products.
            dropout: Optional dropout layer applied to the attention weights.

        Returns:
            A tuple containing:

            - The attended token representations.
            - The attention weights after softmax and optional dropout.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.softmax(dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        return torch.matmul(scores, v), scores

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head self-attention to a token sequence.

        Args:
            x: Input token representations with shape ``(B, N, D)``, where
                `B` is the batch size, `N` is the sequence length, and `D`
                is `embed_dim`.

        Returns:
            Output token representations with shape ``(B, N, D)``.

        Note:
            The attention weights are stored in `attention_score` after the
            forward pass.
        """
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
