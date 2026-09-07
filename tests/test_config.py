import pytest

from vit.config import ViTConfig


def test_vit_config_attributes():
    """Tests if all expected attributes exist and have the correct data types."""
    config = ViTConfig()
    expected_types = {
        "patch_size": int,
        "embed_dim": int,
        "head_size": int,
        "depth": int,
        "mlp_ratio": float,
        "mlp_drop": float,
        "patch_drop": float,
        "attn_proj_drop": float,
        "max_path_drop": float,
    }
    for attr_name, expected_type in expected_types.items():
        assert hasattr(config, attr_name), (
            f"Attribute '{attr_name}' is missing from ViTConfig."
        )
        attr_value = getattr(config, attr_name)
        assert isinstance(attr_value, expected_type), (
            f"Attribute '{attr_name}' has type {type(attr_value).__name__}, "
            f"expected {expected_type.__name__}."
        )
    # Special case for attn_drop since it allows float | None
    assert hasattr(config, "attn_drop"), "Attribute 'attn_drop' is missing."
    attn_drop_val = getattr(config, "attn_drop")
    assert isinstance(attn_drop_val, (float, type(None))), (
        f"Attribute 'attn_drop' has type {type(attn_drop_val).__name__}, "
        "expected float or NoneType."
    )


@pytest.mark.parametrize("method_name", ["tiny", "base", "large", "xlarge"])
def test_vit_config_classmethods(method_name):
    """Tests if the class methods successfully instantiate a ViTConfig object."""
    method = getattr(ViTConfig, method_name)
    config = method()
    assert isinstance(config, ViTConfig), (
        f"ViTConfig.{method_name}() did not return a ViTConfig instance."
    )


@pytest.mark.parametrize(
    "kwargs, match_error",
    [
        ({"patch_size": 0}, "patch_size must be positive"),
        ({"embed_dim": 0}, "embed_dim must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
        ({"embed_dim": 700, "head_size": 64}, "must be exactly divisible"),
        ({"depth": 0}, "depth must be positive"),
        ({"mlp_ratio": 0.0}, "mlp_ratio must be positive"),
        ({"mlp_drop": -0.1}, "mlp_drop must be in"),
        ({"mlp_drop": 1.5}, "mlp_drop must be in"),
        ({"patch_drop": -0.1}, "patch_drop must be in"),
        ({"patch_drop": 1.0}, "patch_drop must be in"),
        ({"attn_proj_drop": 1.2}, "attn_proj_drop must be in"),
        ({"attn_drop": -0.1}, "attn_drop must be None or in"),
        ({"max_path_drop": 1.2}, "max_path_drop must be in"),
    ],
)
def test_vit_config_validation(kwargs, match_error):
    """Ensure invalid parameter values trigger ValueError during instantiation."""
    with pytest.raises(ValueError, match=match_error):
        ViTConfig(**kwargs)
