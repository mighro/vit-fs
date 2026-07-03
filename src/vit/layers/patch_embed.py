from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformers."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        image_size: tuple[int, int],
        embed_dim: int,
    ) -> None:
        """Patch embedding layer for Vision Transformers.

        Args:
            in_channels: Number of input channels.
            patch_size: Size of each patch.
            image_size: Size of the input image.
            embed_dim: Dimension of the embedding.
        """
        super().__init__()
        self.patch_size = patch_size

        self.patching_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 16 != 0 or x.shape[-2] % 16 != 0:
            raise RuntimeError("Input width/height must be divisible by 16")

        return self.flatten(self.patching_conv(x)).permute(0, 2, 1)
