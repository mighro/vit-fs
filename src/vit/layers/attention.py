import torch
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


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over a sequence of tokens.

    Args:
        embed_dim: Width of each token vector. Split evenly across heads.
        head_size: Width of each head's subspace. The head count is
            ``embed_dim // head_size``.
        proj_drop: Dropout probability applied after the output projection.
        attn_drop: Dropout probability applied to the attention weights.
            If `None`, the rate is derived from the head count.

    Raises:
        ValueError: If `embed_dim` is not evenly divisible by `head_size`.

    Attributes:
        attention_score: Softmaxed attention weights from the latest forward
            pass, with shape ``(B, num_heads, N, N)``. `None` before the
            first forward pass.
    """

    # Intuition: a patch embedding only describes one image patch in isolation.
    # Self-attention lets each token ask "which other tokens are relevant to
    # me?" and rebuild itself as a weighted mix of all tokens. A background
    # patch can pick up context from a foreground patch, and vice versa.
    #
    # Per head, each token is mapped to three roles: a Query ("what am I
    # looking for?"), a Key ("what do I offer?"), and a Value ("what do I
    # contribute?"). Dot products between every query and every key give a
    # score matrix showing how much token i cares about token j. Scores are
    # scaled by 1 / sqrt(d_k), normalized row-wise with softmax so each row
    # sums to 1, and used as mixing weights over the values:
    #
    #     Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    #
    # Multiple heads let the model capture several kinds of relationships at
    # once: one head may track texture, another position, another object
    # boundaries. Heads attend independently, are concatenated back together,
    # and mixed by the output projection so they can interact again.
    #
    # Everything is written out in plain PyTorch with no fused kernels. The
    # weights from the latest forward pass are kept on `attention_score` as a
    # learning aid, so you can inspect or plot what each head looked at.

    def __init__(
        self,
        embed_dim: int,
        head_size: int = HEAD_SIZE,
        proj_drop: float = PROJ_DROP,
        attn_drop: float | None = None,
    ):
        """Initialize the attention projections and regularization.

        Args:
            embed_dim: Width of each input/output token vector.
            head_size: Width of each head's Q/K/V subspace.
            proj_drop: Dropout rate after the output projection.
            attn_drop: Dropout rate on the attention weights, or `None` to
                derive one from the head count.

        Raises:
            ValueError: If `embed_dim` is not evenly divisible by `head_size`.
        """
        super().__init__()

        # The embedding is split evenly across heads, so embed_dim must be
        # an exact multiple of head_size (e.g. 768 = 12 heads x 64 dims).
        if embed_dim % head_size != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be exactly divisible "
                f"by head_size ({head_size})."
            )

        self.n_heads = embed_dim // head_size
        self.d_k = head_size
        # Scale factor 1/sqrt(d_k): without it, dot products grow with head
        # size, pushing softmax into saturated regions where gradients vanish.
        self.scale = self.d_k**-0.5

        # One fused QKV projection instead of three separate linears. It
        # computes Q, K, V in a single matmul and we split the result apart
        # in forward(). Mathematically identical, just fewer ops to launch.
        self.w_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # Output projection mixes the concatenated heads back together, so
        # heads can interact after attending independently.
        self.w_o = nn.Linear(embed_dim, embed_dim)

        # Pick a sensible default when the caller does not specify one.
        # More heads -> slightly stronger regularization on the weights.
        # Dropout layers are always constructed; they become no-ops in eval
        # mode.
        if attn_drop is None:
            attn_drop = ATTN_DROP_BASE * (
                1.0 + ATTN_DROP_HEAD_SLOPE * (self.n_heads - ATTN_DROP_REF_HEADS)
            )
            attn_drop = max(ATTN_DROP_MIN, min(ATTN_DROP_MAX, attn_drop))

        self.attn_drop_rate = attn_drop
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.proj_dropout = nn.Dropout(p=proj_drop)
        # Holds the softmaxed attention weights from the latest forward pass.
        # Useful for visualization; not part of the returned output.
        self.attention_score: Tensor | None = None

    @staticmethod
    def attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scale: float,
        dropout: nn.Dropout | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Args:
            q: Queries, ``(B, num_heads, N, d_k)``.
            k: Keys, ``(B, num_heads, N, d_k)``.
            v: Values, ``(B, num_heads, N, d_k)``.
            scale: Scaling factor applied to the dot products.
            dropout: Dropout applied to the attention weights. No-op in eval
                mode.

        Returns:
            A tuple containing:

            - The attended values, ``(B, num_heads, N, d_k)``.
            - The attention weights, ``(B, num_heads, N, N)``. Each row sums
              to 1, aside from dropout rescaling.
        """
        # Q @ K^T compares every query position against every key position:
        # scores[b, h, i, j] is how much token i attends to token j in head h.
        # The transpose turns per-token vectors into that pairwise table.
        #
        # Dividing by sqrt(d_k) keeps dot products in a range where softmax
        # stays sensitive. Without it, larger head dims produce larger
        # magnitudes, softmax saturates toward one-hot, and gradients vanish.
        # Softmax over the last dim turns each row into mixing proportions,
        # and multiplying by V rebuilds every token as that weighted blend.
        # Step 1: pairwise similarity scores. For every query position i and
        # key position j, scores[..., i, j] measures how much token i wants
        # to look at token j. Shape: (B, num_heads, N, N).
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Step 2: normalize each row into a probability distribution. After
        # this, scores[..., i, :] sums to 1 and can be read as "what fraction
        # of token i's new representation comes from each position".
        scores = scores.softmax(dim=-1)

        # Step 3 (training only): randomly drop some attention links. This is
        # a no-op in eval mode, like all nn.Dropout layers.
        if dropout is not None:
            scores = dropout(scores)

        # Step 4: weighted sum of values. Each output token is a convex-ish
        # combination of all value vectors, weighted by its attention row.
        return torch.matmul(scores, v), scores

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention to a token sequence.

        Args:
            x: Token vectors, ``(B, N, D)``, where `B` is the batch size,
                `N` is the sequence length (patches plus `[CLS]`), and `D`
                is `embed_dim`.

        Returns:
            Context-enriched tokens with shape ``(B, N, D)``.

        Note:
            The attention weights from this pass are stored on
            `attention_score` with shape ``(B, num_heads, N, N)``.
        """
        B, N, _ = x.shape

        # Step 1: fused QKV projection + split into heads.
        # w_qkv(x) has shape (B, N, 3 * D): each token's query, key, and value
        # side by side. Reshape to expose the Q/K/V axis and the per-head
        # subspaces, then permute so the Q/K/V axis comes first and can be
        # unpacked directly. Every head then sees its own thin slice of every
        # token:
        #   (B, N, 3, num_heads, d_k) -> (3, B, num_heads, N, d_k)
        q, k, v = (
            self.w_qkv(x)
            .reshape(B, N, 3, self.n_heads, self.d_k)
            .permute(2, 0, 3, 1, 4)
        )

        # Step 2: run the explicit attention computation defined above.
        # `out` holds per-head attended values; the weights are saved on the
        # module so educational callers can visualize them afterwards.
        out, self.attention_score = self.attention(
            q, k, v, self.scale, self.attn_dropout
        )

        # Step 3: merge heads back together. Transpose to (B, N, num_heads,
        # d_k) and flatten the last two dims to recover (B, N, D), where
        # D = num_heads * d_k = embed_dim.
        out = out.transpose(1, 2).contiguous().view(B, N, self.n_heads * self.d_k)

        # Step 4: final linear mixing across heads + dropout.
        return self.proj_dropout(self.w_o(out))
