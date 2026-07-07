import pytest
import torch

from vit.config import ViTConfig
from vit.model import ClassificationViT


@pytest.fixture
def sample_config():
    """Fixture providing a small config for faster testing."""
    return ViTConfig.tiny()


@pytest.fixture
def sample_model(sample_config):
    """Fixture providing an initialized ClassificationViT model."""
    return ClassificationViT(
        in_channels=3, image_size=(224, 224), num_classes=10, config=sample_config
    )


def test_classification_vit_initialization(sample_model, sample_config):
    """Tests if the model initializes with the correct structural attributes."""
    assert isinstance(sample_model, ClassificationViT)
    assert sample_model.num_classes == 10
    assert sample_model.in_channels == 3
    assert sample_model.image_size == (224, 224)
    assert len(sample_model.blocks) == sample_config.depth


def test_classification_vit_forward(sample_model):
    """Tests the forward pass to ensure output shapes and types are correct."""
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    output = sample_model(dummy_input)

    assert output.shape == (batch_size, 10), (
        f"Expected shape {(batch_size, 10)}, got {output.shape}"
    )
    assert output.dtype == torch.float32


def test_classification_vit_inference(sample_model):
    """Tests the inference wrapper to ensure it returns normalized probabilities."""
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    output = sample_model.inference(dummy_input)

    assert output.shape == (batch_size, 10)

    sums = output.sum(dim=-1)

    assert torch.allclose(sums, torch.ones_like(sums)), (
        "Inference outputs do not sum to 1.0"
    )

    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
        "Inference outputs are not bounded between 0 and 1"
    )

    assert not sample_model.training, (
        "Model is not in eval mode after calling inference()"
    )
