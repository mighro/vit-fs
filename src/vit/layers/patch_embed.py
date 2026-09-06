from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of patch embeddings.

    The image is divided into non-overlapping patches using a convolution
    whose kernel size and stride are both equal to `patch_size`. Each patch
    is projected directly into the transformer embedding dimension.

    For an image of size ``(H, W)``, the resulting number of patches is:

        (H / patch_size) * (W / patch_size)

    The convolution output is then flattened into a sequence of token
    embeddings.

    Args:
        in_channels: Number of channels in the input image.
        patch_size: Height and width of each square image patch.
        image_size: Expected input image dimensions as ``(height, width)``.
        embed_dim: Dimension of each output patch embedding.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        image_size: tuple[int, int],
        embed_dim: int,
    ) -> None:
        """Initialize the image-to-patch embedding layer.

        Args:
            in_channels: Number of channels in the input image.
            patch_size: Height and width of each square patch.
            image_size: Expected input image dimensions as ``(height, width)``.
            embed_dim: Dimension of each output patch embedding.
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
        """Convert an image batch into a sequence of patch tokens.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)``.

        Returns:
            Patch embeddings with shape ``(B, N, D)``, where `N` is the number
            of image patches and `D` is `embed_dim`.

        Raises:
            RuntimeError: If either input image dimension is not divisible by
                `patch_size`.
        """
        if x.shape[-1] % self.patch_size != 0 or x.shape[-2] % self.patch_size != 0:
            raise RuntimeError(
                f"Input width/height must be divisible by {self.patch_size}"
            )

        return self.flatten(self.patching_conv(x)).permute(0, 2, 1)
