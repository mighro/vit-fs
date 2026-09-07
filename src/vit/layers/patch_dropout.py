import torch
from torch import Tensor, nn


class PatchDropout(nn.Module):
    """Randomly drop patch tokens during training.

    Unlike standard dropout (:class:`torch.nn.Dropout`), which zeroes individual
    feature elements, patch dropout discards entire token vectors from the
    sequence. This regularizes the network, reduces compute during training,
    and encourages representations robust to missing visual context.

    During evaluation or when `prob == 0.0`, the input sequence is returned
    unchanged.

    Note:
        The classification (``[CLS]``) token should be prepended *after*
        applying patch dropout so that it is never discarded.

    Args:
        prob: Probability of dropping patch tokens. Must be in the range
            ``[0.0, 1.0)``. Defaults to 0.0.

    Raises:
        ValueError: If `prob` is not strictly in the range ``[0.0, 1.0)``.
    """

    def __init__(self, prob: float = 0.0) -> None:
        """Initialize the PatchDropout module.

        Args:
            prob: Probability of dropping patch tokens. Defaults to 0.0.

        Raises:
            ValueError: If `prob` is not strictly in the range ``[0.0, 1.0)``.
        """
        super().__init__()
        if not 0.0 <= prob < 1.0:
            raise ValueError("Patch dropout probability must be in [0, 1).")
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        """Randomly subsample patch tokens along the sequence dimension.

        Args:
            x: Input patch token representations with shape ``(B, N, D)``,
                where ``B`` is the batch size, ``N`` is the number of image
                patches, and ``D`` is the embedding dimension.

        Returns:
            Subsampled token representations with shape ``(B, N', D)`` during
            training (where ``N' = max(1, int(N * (1 - prob)))``), or the
            original tensor ``(B, N, D)`` during evaluation.
        """
        if not self.training or self.prob == 0.0:
            return x

        B, N, D = x.shape
        keep_prob = 1.0 - self.prob
        keep = max(1, int(N * keep_prob))

        noise = torch.rand(B, N, device=x.device)
        ids_keep = noise.topk(keep, dim=1, largest=True).indices
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)

        return torch.gather(x, dim=1, index=ids_keep)
