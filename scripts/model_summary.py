"""Generate torchinfo model summaries and comparative statistics for all ViT configs."""

import sys

from vit import ClassificationViT, ViTConfig

try:
    from torchinfo import summary
except ImportError:
    print("Error: 'torchinfo' is not installed.")
    sys.exit(1)


def main() -> None:
    image_size = (224, 224)
    in_channels = 3
    num_classes = 1000
    batch_size = 1

    configs: dict[str, ViTConfig] = {
        "tiny": ViTConfig.tiny(),
        "base": ViTConfig.base(),
        "large": ViTConfig.large(),
        "xlarge": ViTConfig.xlarge(),
    }

    stats_summary: list[dict[str, str | int | float]] = []

    print("\n" + "=" * 90)
    print(" Vision Transformer (ViT) Architecture Summaries via torchinfo")
    print(
        f" Input shape: ({batch_size}, {in_channels}, {image_size[0]}, {image_size[1]}) | Classes: {num_classes}"
    )
    print("=" * 90 + "\n")

    for name, config in configs.items():
        print("*" * 90)
        print(f" CONFIGURATION: {name.upper()}")
        print(
            f" embed_dim={config.embed_dim}, heads={config.embed_dim // config.head_size}, "
            f"depth={config.depth}, mlp_ratio={config.mlp_ratio:.2f}"
        )
        print("*" * 90)

        model = ClassificationViT(
            in_channels=in_channels,
            image_size=image_size,
            num_classes=num_classes,
            config=config,
        )
        model.eval()

        model_stats = summary(
            model,
            input_size=(batch_size, in_channels, *image_size),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=3,
            verbose=1,
        )
        print("\n")

        total_params = model_stats.total_params
        total_mult_adds = getattr(model_stats, "total_mult_adds", 0)
        param_bytes = getattr(model_stats, "total_param_bytes", 0)
        size_mb = (
            param_bytes / (1024**2) if param_bytes else (total_params * 4) / (1024**2)
        )

        stats_summary.append(
            {
                "config": name,
                "embed_dim": config.embed_dim,
                "heads": config.embed_dim // config.head_size,
                "depth": config.depth,
                "params_m": f"{total_params / 1e6:.2f}M",
                "mult_adds_g": f"{total_mult_adds / 1e9:.2f}G",
                "size_mb": f"{size_mb:.2f} MB",
            }
        )

    # Print consolidated comparative table
    print("=" * 90)
    print(" COMPARATIVE SUMMARY")
    print("=" * 90)
    header = (
        f"{'Config':<10} | {'Embed Dim':<10} | {'Heads':<6} | {'Depth':<6} | "
        f"{'Params (M)':<12} | {'Mult-Adds (G)':<14} | {'Model Size':<10}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)
    for row in stats_summary:
        print(
            f"{row['config']:<10} | {row['embed_dim']:<10} | {row['heads']:<6} | {row['depth']:<6} | "
            f"{row['params_m']:<12} | {row['mult_adds_g']:<14} | {row['size_mb']:<10}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
