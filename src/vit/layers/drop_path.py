from torch import Tensor, nn


def get_drop_rate(idx: int, depth: int, max_drop_rate: float) -> float:
    """Compute the stochastic-depth rate for a transformer block.

    The drop rate increases linearly with the block's depth. The first
    block has a rate of 0, while the final block reaches `max_drop_rate`.

    Args:
        idx: Zero-based index of the transformer block.
        depth: Total number of transformer blocks.
        max_drop_rate: Maximum drop probability assigned to the final block.

    Returns:
        The DropPath probability for the block at `idx`.

    Raises:
        ValueError: If `idx` is greater than or equal to `depth`.
    """
    if depth <= 1:
        return 0.0
    if idx >= depth:
        raise ValueError(f"idx ({idx}) must be strictly less than depth ({depth})")

    return max_drop_rate * (idx / (depth - 1))


class DropPath(nn.Module):
    """Apply stochastic depth to a residual branch.

    During training, entire residual paths are randomly dropped on a
    per-sample basis. The surviving paths are scaled by
    ``1 / (1 - drop_prob)`` so that their expected value remains unchanged.

    During evaluation, DropPath acts as an identity function.

    Args:
        drop_prob: Probability of dropping the residual path.
    """

    def __init__(self, drop_prob: float = 0.0):
        """Initialize a DropPath module.

        Args:
            drop_prob: Probability of dropping the residual path. Defaults to 0.0.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        shape = (x.shape[0], 1, 1)
        mask = x.new_empty(shape).bernoulli_(self.keep_prob)

        return x.mul(mask / self.keep_prob)
