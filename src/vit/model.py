import torch
from torch import Tensor, nn

from .config import ViTConfig
from .layers import LayerNormalisation, get_drop_rate
from .modules import EncoderBlock, ViTEmbeddings


class ClassificationViT(nn.Module):
    """
    Vision Transformer (ViT) architecture for image classification.

    This module implements an encoder-only transformer that processes
    image patches and uses a prepended class token ([CLS]) to perform
    classification tasks.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: tuple[int, int],
        num_classes: int,
        config: ViTConfig,
    ):
        """
        Initializes the ClassificationViT model.

        Args:
            in_channels (int): Number of color channels in the input images (e.g., 3 for RGB).
            image_size (tuple[int, int]): The height and width of the input images.
            num_classes (int): The number of output classes for classification.
            config (ViTConfig): Configuration object containing transformer hyperparameters
                (e.g., depth, embed_dim, patch_size, dropout rates).
        """
        super().__init__()

        self.config = config
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_size = image_size

        self.embeddings = ViTEmbeddings(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            patch_dropout_rate=config.patch_drop,
        )

        dpr = [
            get_drop_rate(i, config.depth, config.max_path_drop)
            for i in range(config.depth)
        ]
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=config.embed_dim,
                    head_size=config.head_size,
                    mlp_ratio=config.mlp_ratio,
                    mlp_drop=config.mlp_drop,
                    proj_drop=config.attn_proj_drop,
                    drop_path=dpr[i],
                    attn_drop=config.attn_drop,
                )
                for i in range(config.depth)
            ]
        )

        self.classification_head = nn.Sequential(
            LayerNormalisation(config.embed_dim),
            nn.Linear(config.embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass yielding raw classification logits."""
        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x)

        cls_token = x[:, 0]

        return self.classification_head(cls_token)

    @torch.inference_mode()
    def inference(self, x: Tensor) -> Tensor:
        """
        Performs model inference, returning class probabilities.

        Automatically sets the model to evaluation mode and applies
        softmax to the raw logits.

        Args:
            x (Tensor): A batch of input images with shape `(B, in_channels, H, W)`.

        Returns:
            Tensor: Normalized class probabilities of shape `(B, num_classes)`.
        """
        self.eval()
        return torch.softmax(self.forward(x), dim=-1)
