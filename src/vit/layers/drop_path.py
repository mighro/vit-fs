from torch import Tensor, nn


def get_drop_rate(idx: int, depth: int, max_drop_rate: float) -> float:
    """Calculates the linear drop path rate for a given layer index."""
    if depth <= 1:
        return 0.0
    if idx >= depth:
        raise ValueError(f"idx ({idx}) must be strictly less than depth ({depth})")

    return max_drop_rate * (idx / (depth - 1))


class DropPath(nn.Module):
    """
    Implements Stochastic Depth (DropPath).

    Randomly zeroes out elements (per sample in a batch) during training with
    probability `drop_prob`, scaling the remaining outputs by `1 / (1 - drop_prob)`
    to maintain expected values. Acts as an identity layer during evaluation.
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Initializes the DropPath module.

        Args:
            drop_prob (float): The probability of dropping a path. Default: 0.0.
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
