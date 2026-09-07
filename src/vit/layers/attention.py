import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

HEAD_SIZE = 64
PROJ_DROP = 0.1

# Attention Dropout Heuristic Constants
#
# When `attn_drop` is not explicitly provided (`None`), the dropout rate is
# scaled dynamically with the number of attention heads:
#   rate = BASE * (1 + HEAD_SLOPE * (n_heads - REF_HEADS))
#
# Models with more attention heads have higher representational capacity and
# benefit from slightly stronger regularization on the attention weights.
# The resulting rate is clamped between MIN and MAX bounds for stability.
ATTN_DROP_BASE: float = 0.1  # Baseline dropout rate at reference head count
ATTN_DROP_HEAD_SLOPE: float = 0.05  # Linear scaling step per head offset
ATTN_DROP_REF_HEADS: int = 8  # Pivot head count where scaling multiplier is 1.0
ATTN_DROP_MIN: float = 0.05  # Lower bound for auto-calculated attention dropout
ATTN_DROP_MAX: float = 0.2  # Upper bound for auto-calculated attention dropout

# Attention backend selection
#
# "torch":
#   Uses PyTorch's optimized scaled_dot_product_attention.
#   On supported hardware this automatically uses FlashAttention kernels.
#
# "manual":
#   Uses explicit QK^T attention computation.
#   Useful for educational purposes and attention visualization.
ATTENTION_BACKEND: str = "torch"


class MultiHeadSelfAttention(nn.Module):
    """Compute multi-head self-attention over a sequence of tokens.

    Each input token is projected into a Query, Key, and Value representation.
    The queries and keys are used to calculate scaled attention scores, which
    determine how strongly each token attends to every other token.

    The embedding dimension is divided across multiple attention heads.
    Each head performs attention independently before the results are
    concatenated and projected back to the original embedding dimension.

    When using ``backend="manual"``, the computed attention weights are stored
    in `attention_score` after each forward pass. Under ``backend="torch"``,
    weights are fused and not materialized, so `attention_score` is set to `None`.

    Args:
        embed_dim: Dimension of each input and output token representation.
        head_size: Dimension of each individual attention head.
        proj_drop: Dropout probability applied after the output projection.
        attn_drop: Dropout probability applied to the attention weights.
            If `None`, a rate is selected based on the number of heads.
        backend: Attention computation backend, either ``"torch"`` or
            ``"manual"``. Defaults to `ATTENTION_BACKEND`.

    Raises:
        ValueError: If `embed_dim` is not evenly divisible by `head_size`,
            or if `backend` is not one of ``{"torch", "manual"}``.
    """

    def __init__(
        self,
        embed_dim: int,
        head_size: int = HEAD_SIZE,
        proj_drop: float = PROJ_DROP,
        attn_drop: float | None = None,
        backend: str = ATTENTION_BACKEND,
    ):
        """Initialize the multi-head self-attention module.

        Args:
            embed_dim: Dimension of each input and output token representation.
            head_size: Dimension of each attention head.
            proj_drop: Dropout probability applied after the output projection.
            attn_drop: Dropout probability applied to the attention weights.
                If `None`, the dropout rate is determined automatically.
            backend: Attention implementation backend, either ``"torch"`` or
                ``"manual"``. Defaults to `ATTENTION_BACKEND`.

        Raises:
            ValueError: If `embed_dim` is not evenly divisible by `head_size`,
                or if `backend` is invalid.
        """
        super().__init__()

        if embed_dim % head_size != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be exactly divisible "
                f"by head_size ({head_size})."
            )

        if backend not in ("torch", "manual"):
            raise ValueError(
                f"Invalid attention backend '{backend}'. Expected 'torch' or 'manual'."
            )

        self.backend = backend
        self.n_heads = embed_dim // head_size
        self.d_k = head_size
        self.scale = self.d_k**-0.5

        self.w_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim)

        if attn_drop is None:
            attn_drop = ATTN_DROP_BASE * (
                1.0 + ATTN_DROP_HEAD_SLOPE * (self.n_heads - ATTN_DROP_REF_HEADS)
            )
            attn_drop = max(ATTN_DROP_MIN, min(ATTN_DROP_MAX, attn_drop))

        self.attn_drop_rate = attn_drop
        self.attn_dropout = nn.Dropout(p=attn_drop) if backend == "manual" else None
        self.proj_dropout = nn.Dropout(p=proj_drop)
        self.attention_score: Tensor | None = None

    @staticmethod
    def attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scale: float,
        dropout: nn.Dropout | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute scaled dot-product attention explicitly.

        Args:
            q: Query tensor with shape ``(B, num_heads, N, d_k)``.
            k: Key tensor with shape ``(B, num_heads, N, d_k)``.
            v: Value tensor with shape ``(B, num_heads, N, d_k)``.
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
            When `backend="manual"`, attention weights are stored in
            `attention_score`. Under `backend="torch"`, weights are fused and
            `attention_score` is set to `None`.
        """
        B, N, _ = x.shape

        q, k, v = (
            self.w_qkv(x)
            .reshape(B, N, 3, self.n_heads, self.d_k)
            .permute(
                2, 0, 3, 1, 4
            )  # (B, N, 3, num_heads, d_k) -> (3, B, num_heads, N, d_k)
        )

        if self.backend == "torch":
            dropout_p = self.attn_drop_rate if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=self.scale,
                dropout_p=dropout_p,
            )
            self.attention_score = None
        else:
            out, self.attention_score = self.attention(
                q, k, v, self.scale, self.attn_dropout
            )

        out = out.transpose(1, 2).contiguous().view(B, N, self.n_heads * self.d_k)

        return self.proj_dropout(self.w_o(out))
