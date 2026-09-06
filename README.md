# vit-fs

An implementation of a Vision Transformer (ViT) from scratch using PyTorch.

The project is mainly for learning and understanding how the different parts of a ViT work by implementing them directly rather than using an existing high-level implementation.

## What is implemented

* Patch embedding using a strided convolution
* Learnable `[CLS]` token
* Learnable positional embeddings
* Multi-head self-attention
* Transformer encoder blocks
* Layer normalization
* SwiGLU feed-forward layers
* Dropout
* Stochastic depth / DropPath
* Classification head
* Configurable ViT architecture
* Unit tests for the individual components and model

The main classes are:

```python
from vit import ClassificationViT, ViTConfig
```

## Project structure

```text
.
├── src/
│   └── vit/
│       ├── layers/
│       │   ├── attention.py
│       │   ├── cls_token.py
│       │   ├── drop_path.py
│       │   ├── layer_norm.py
│       │   ├── mlp.py
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

These are project-specific configurations and are not exact reproductions of the configurations from the original ViT paper.

The configuration includes:

* Patch size
* Embedding dimension
* Attention head size
* Number of transformer blocks
* Feed-forward expansion ratio
* Dropout rates
* Stochastic depth rate

A configuration can also be created manually:

```python
config = ViTConfig(
    patch_size=16,
    embed_dim=384,
    head_size=64,
    depth=12,
)
```

## Design

The model follows this general pipeline:

```text
Image
  │
  ▼
Patch Embedding
  │
  ▼
Patch Dropout
  │
  ▼
Add Positional Embeddings
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

## Project status

This is a learning project.

The code is kept relatively low-level so that the individual parts of the ViT can be inspected, tested, and changed independently.

It is not intended to replace optimized ViT implementations or to be a production training framework.
