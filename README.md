# vit-fs

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Vision Transformer built from scratch in PyTorch, one layer at a time.

The goal here wasn't to reproduce Google's 2020 ViT paper exactly. It was to understand how the pieces actually work by writing them myself: attention, positional embeddings, the encoder stack, the whole pipeline. Along the way I pulled in some design choices from more recent transformer literature that I found interesting enough to experiment with.

## What's in here

Everything you'd expect from a ViT, plus a few modern twists:

- **Patch embedding** via strided conv (with shape validation so you can't accidentally feed it the wrong image size)
- **Learnable `[CLS]` token** and **positional embeddings**
- **Multi-head self-attention** with a dynamic dropout heuristic (more heads, slightly more dropout, clamped to a sane range)
- **SwiGLU feed-forward network** instead of the original two-linear + ReLU
- **Patch dropout** — drops whole token vectors, not individual features
- **Stochastic depth (DropPath)** with a linear schedule across blocks
- **Two attention backends**: PyTorch's fused `scaled_dot_product_attention` for speed, and a manual QK^T version if you want to inspect the attention matrices yourself
- **Four config presets** (`tiny`, `base`, `large`, `xlarge`) with validation on construction
- **Unit tests** for each layer and the full model

The public API is just two things:

```python
from vit import ClassificationViT, ViTConfig
```

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mighro/vit-fs.git
cd vit-fs

uv venv --python 3.12
uv sync
```

Then:

```python
import torch
from vit import ClassificationViT, ViTConfig

model = ClassificationViT(
    in_channels=3,
    image_size=(224, 224),
    num_classes=10,
    config=ViTConfig.tiny(),
)

logits = model(torch.randn(2, 3, 224, 224))
print(logits.shape)  # torch.Size([2, 10])

# Or get probabilities directly:
probs = model.inference(torch.randn(2, 3, 224, 224))
```

`inference()` wraps the forward pass with `torch.inference_mode()`, switches the model to eval mode (with a warning if you call it while training), and applies softmax. Handy for quick checking without setting up a separate eval loop.

## Model presets

| Config | Embed Dim | Heads | Depth | Params |
|--------|-----------|-------|-------|--------|
| `tiny` | 256 | 4 | 10 | 9.04M |
| `base` | 384 | 6 | 12 | 22.01M |
| `large` | 512 | 8 | 14 | 48.75M |
| `xlarge` | 768 | 12 | 16 | 114.82M |

*(Input: `(1, 3, 224, 224)`, 1000 classes)*

These are my own presets, not the exact specs from the original paper. All values get validated on construction — you'll get a clear `ValueError` if `embed_dim` isn't divisible by `head_size`, or a dropout rate goes out of range, etc.

You can also build a custom config:

```python
config = ViTConfig(patch_size=16, embed_dim=384, head_size=64, depth=12)
```

Run `uv run python scripts/model_summary.py` to regenerate the table above with per-layer detail via `torchinfo`.

## How the forward pass works

```
Image (B, C, H, W)
│
▼
Patch Embedding ──── conv2d with kernel=stride=patch_size
│
▼
Positional Embedding ── add learnable vector per token position
│
▼
Patch Dropout ─────── randomly discard whole tokens (training only)
│
▼
Prepend [CLS] ─────── learnable token, never dropped
│
▼
N × Encoder Block ──── pre-norm: LayerNorm → Attention → residual
│                     pre-norm: LayerNorm → SwiGLU FFN → residual
│                     (both residuals pass through DropPath)
│
▼
Take [CLS] ────────── first token of the sequence
│
▼
Classification Head ─ LayerNorm → Linear
│
▼
Logits (B, num_classes)
```

A few notes on the choices:

**Pre-norm over post-norm.** Normalization happens before each sublayer, not after. This tends to stabilize training for deeper stacks.

**SwiGLU instead of ReLU MLP.** The feed-forward network uses a gated formulation: `SiLU(W_gate(x)) * W_up(x)`, then projects back down. The hidden dim is rounded up to a multiple of 256 for hardware-friendly tensor shapes.

**Dynamic attention dropout.** If you don't specify `attn_drop` in the config, it's computed as `0.1 * (1 + 0.05 * (n_heads - 8))`, clamped to `[0.05, 0.2]`. The idea: bigger models with more heads benefit from a touch more regularization on the attention weights.

**Two attention backends.** The default uses `F.scaled_dot_product_attention`, which picks FlashAttention kernels automatically when available. Set `backend="manual"` on `MultiHeadSelfAttention` if you want the raw attention scores accessible in `attention_score` for visualization or debugging.

## Repo layout

```
src/vit/
├── layers/          # Individual building blocks
│   ├── attention.py       # MultiHeadSelfAttention
│   ├── cls_token.py       # get_cls_token()
│   ├── drop_path.py       # DropPath + get_drop_rate()
│   ├── layer_norm.py      # LayerNormalisation (hand-written, not nn.LayerNorm)
│   ├── mlp.py             # FeedForward (SwiGLU)
│   ├── patch_dropout.py   # PatchDropout
│   ├── patch_embed.py     # PatchEmbedding
│   └── pos_embed.py       # PosEmbedding
├── modules.py         # ViTEmbeddings (input pipeline), EncoderBlock
├── model.py           # ClassificationViT (the full model)
└── config.py          # ViTConfig dataclass

scripts/model_summary.py   # torchinfo-based model stats
tests/                     # pytest suite for all layers + end-to-end
```

Layer norm is written out explicitly rather than wrapping `torch.nn.LayerNorm` — I wanted to see the mean/variance/scale/bias operations happen, not hide them behind a library call. Same spirit for everything else: if there's a non-obvious step, it's visible in the code.

## Testing

```bash
uv run pytest
```

Covers shape checks, gradient flow, edge cases (non-divisible dims, invalid backends, eval vs train behavior), and the inference-mode transition.

## What LLMs helped with

English isn't my first language, so the prose, README, docstrings, general wording, is LLM output that I edited until it stopped sounding like LLM output. The code is mine.

The tests aren't. I wanted a test suite and kept not writing one, so the model wrote them instead. They pass and they cover what the Testing section claims, but keep in mind they came from the same tool that wrote this sentence. 

## License

MIT. Do whatever you want with it, no strings attached.
