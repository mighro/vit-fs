import torch

from vit.layers.drop_path import DropPath


def test_drop_path_no_drop_probability():
    """Test that drop_prob=0.0 returns the input unchanged."""
    drop_path = DropPath(drop_prob=0.0)
    x = torch.randn(64, 3, 4)
    out = drop_path(x)
    assert torch.allclose(out, x)


def test_drop_path_eval_mode():
    """Test that DropPath acts as an identity layer during evaluation."""
    drop_path = DropPath(drop_prob=0.5)
    drop_path.eval()
    x = torch.randn(64, 3, 4)
    out = drop_path(x)
    assert torch.allclose(out, x)


def test_drop_path_train_mode_shape():
    """Test that output shape matches input shape during training."""
    drop_path = DropPath(drop_prob=0.5)
    drop_path.train()
    x = torch.randn(4, 16, 4)
    out = drop_path(x)
    assert out.shape == x.shape


def test_drop_path_train_mode_values():
    """Test that non-dropped elements are scaled by 1/(1-drop_prob)."""
    torch.manual_seed(42)
    drop_path = DropPath(drop_prob=0.5)
    drop_path.train()
    x = torch.ones(64, 3, 4)
    out = drop_path(x)

    # With drop_prob=0.5, keep_prob=0.5, scaling factor is 2.0
    # Output should only contain 0.0 and 2.0
    unique_vals = torch.unique(out)
    assert set(unique_vals.tolist()) == {0.0, 2.0}


def test_drop_path_high_probability():
    """Test behavior with high drop probability."""
    torch.manual_seed(123)
    drop_path = DropPath(drop_prob=0.9)
    drop_path.train()
    x = torch.ones(100, 1, 1)
    out = drop_path(x)

    # With high drop rate, most should be zeroed out
    assert (out == 0.0).sum() > out.numel() * 0.5
