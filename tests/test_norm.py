import torch

from vit.layers.layer_norm import LayerNormalisation


def test_layer_norm():
    l_norm = LayerNormalisation(128)

    x = torch.randn(1, 190, 128)
    x_norm = l_norm(x)

    assert torch.allclose(x_norm.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(x_norm.std(), torch.tensor(1.0), atol=1e-2)
