import warnings

import torch
from torch import Tensor, nn

from .config import ViTConfig
from .layers import LayerNormalisation, get_drop_rate
from .modules import EncoderBlock, ViTEmbeddings


class ClassificationViT(nn.Module):
    """Vision Transformer encoder for image classification.

    The model converts an input image into a sequence of patch embeddings,
    prepends a learnable classification token, and processes the resulting
    sequence with a stack of Transformer encoder blocks.

    The final representation of the `[CLS]` token is passed through a
    normalization layer and linear classification head to produce one logit
    for each target class.

    The model consists of three main stages:

    1. Image-to-token conversion with `ViTEmbeddings`.
    2. Token processing with Transformer encoder blocks.
    3. Classification from the final `[CLS]` representation.

    Args:
        in_channels: Number of channels in each input image.
        image_size: Expected input image size as ``(height, width)``.
        num_classes: Number of classes predicted by the classification head.
        config: Transformer architecture and regularization configuration.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: tuple[int, int],
        num_classes: int,
        config: ViTConfig,
    ):
        """Initialize the Vision Transformer classifier.

        Args:
            in_channels: Number of channels in each input image.
            image_size: Expected input image dimensions as ``(height, width)``.
            num_classes: Number of output classes.
            config: Configuration containing the transformer architecture and
                regularization hyperparameters.
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
        """Run a forward pass and return classification logits.

        The input image is converted into patch tokens, processed by the
        Transformer encoder, and classified using the final `[CLS]` token.

        This method returns raw logits rather than probabilities. Apply an
        appropriate loss function such as cross-entropy during training, or
        softmax externally when probabilities are required.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)`` or, when supported
                by the embedding layer, ``(C, H, W)``.

        Returns:
            Classification logits with shape ``(B, num_classes)``.
        """
        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x)

        cls_token = x[:, 0]

        return self.classification_head(cls_token)

    @torch.inference_mode()
    def inference(self, x: Tensor) -> Tensor:
        """Run inference and return normalized class probabilities.

        This method disables gradient tracking with `torch.inference_mode()` and
        ensures the model is in evaluation mode before performing the forward pass.

        If the model is currently in training mode, a `UserWarning` is emitted
        because calling this method changes the model's training state. The model
        remains in evaluation mode after inference completes.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)`` or, when supported
                by the embedding layer, ``(C, H, W)``.

        Returns:
            Class probabilities with shape ``(B, num_classes)``. Each row sums
            to approximately 1.

        Warnings:
            UserWarning: If the model is currently in training mode and must be
                switched to evaluation mode.
        """
        if self.training:
            warnings.warn(
                "inference() is switching the model from training mode to "
                "evaluation mode. The model will remain in evaluation mode "
                "after inference().",
                UserWarning,
                stacklevel=2,
            )

        self.eval()
        return torch.softmax(self.forward(x), dim=-1)
