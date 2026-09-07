# vit-fs

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern Vision Transformer (ViT) inspired vision encoder implemented from scratch in PyTorch.

This project is designed for understanding and experimenting with transformer-based vision models by implementing the major components directly rather than relying on high-level libraries.

While inspired by the original Vision Transformer (ViT) architecture, this implementation incorporates several modern transformer design choices including:

- SwiGLU feed-forward networks
- Patch dropout
- Stochastic depth (DropPath)
- Configurable attention backends
- Dynamic attention dropout
- Modular transformer components

## What is implemented

* Patch embedding using a strided convolution with image size validation
* Learnable `[CLS]` token
* Learnable positional embeddings
* Multi-head self-attention with dynamic attention dropout heuristic
* Transformer encoder blocks
* Layer normalization
* SwiGLU feed-forward layers
* Patch dropout and dropout regularization
* Stochastic depth / DropPath
* Classification head
* Configurable ViT architecture with post-init validation
* Model inspection script using `torchinfo`
* Unit tests for individual components and the end-to-end model

The main classes are:

```python
from vit import ClassificationViT, ViTConfig
```

## Project structure

```text
.
├── scripts/
│   └── model_summary.py
├── src/
│   └── vit/
│       ├── layers/
│       │   ├── attention.py
│       │   ├── cls_token.py
│       │   ├── drop_path.py
│       │   ├── layer_norm.py
│       │   ├── mlp.py
│       │   ├── patch_dropout.py
│       │   ├── patch_embed.py
│       │   └── pos_embed.py
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       └── modules.py
├── tests/
├── pyproject.toml
├── README.md
└── .python-version
```

## Usage

Clone the repository:

```bash
git clone https://github.com/mighro/vit-fs.git
cd vit-fs
```

Create the environment and install dependencies:

```bash
uv venv --python 3.12
uv sync
```

## Example

A small model can be created using the `tiny` configuration:

```python
import torch

from vit import ClassificationViT, ViTConfig

config = ViTConfig.tiny()

model = ClassificationViT(
    in_channels=3,
    image_size=(224, 224),
    num_classes=10,
    config=config,
)

x = torch.randn(2, 3, 224, 224)
logits = model(x)

print(logits.shape)
# torch.Size([2, 10])
```

For inference:

```python
probabilities = model.inference(x)
```

## Model configuration

`ViTConfig` provides four predefined configurations:

```python
ViTConfig.tiny()
ViTConfig.base()
ViTConfig.large()
ViTConfig.xlarge()
```

These presets are verified with post-initialization validation to ensure that dimensions (e.g. `embed_dim % head_size == 0`), layer depths, and dropout probabilities are valid.

A configuration can also be created manually:

```python
config = ViTConfig(
    patch_size=16,
    embed_dim=384,
    head_size=64,
    depth=12,
)
```

## Model statistics

Model parameters and computational metrics can be inspected across all presets using the summary script:

```bash
uv run python scripts/model_summary.py
```

### Comparative summary (Input shape: `(1, 3, 224, 224)` | Classes: 1000)

| Config | Embed Dim | Heads | Depth | Params (M) | Mult-Adds (G) | Model Size |
|---|---|---|---|---|---|---|
| `tiny` | 256 | 4 | 10 | 9.04M | 0.05G | 34.47 MB |
| `base` | 384 | 6 | 12 | 22.01M | 0.08G | 83.97 MB |
| `large` | 512 | 8 | 14 | 48.75M | 0.13G | 185.98 MB |
| `xlarge` | 768 | 12 | 16 | 114.82M | 0.23G | 438.00 MB |

## Design

The model follows this general pipeline:

```text
Image
  │
  ▼
Patch Embedding
  │
  ▼
Add Positional Embeddings
  │
  ▼
Patch Dropout
  │
  ▼
Prepend [CLS] Token
  │
  ▼
Transformer Encoder Blocks
  │
  ├── LayerNorm
  ├── Multi-Head Self-Attention
  ├── Residual Connection
  ├── LayerNorm
  ├── SwiGLU Feed-Forward Network
  └── Residual Connection
  │
  ▼
[CLS] Token
  │
  ▼
Classification Head
  │
  ▼
Class Logits
```

## LLM assistance

LLMs were used during development, mainly for writing and language assistance. English is not my first language, so I use them to help with English writing.

They were used for:

* README writing and editing
* Docstrings
* Unit tests
* Grammar and wording

The implementation was written by me. I review and modify suggestions before using them in the project.

## License

This project is licensed under the MIT License.
