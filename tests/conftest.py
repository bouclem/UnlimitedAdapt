"""Test fixtures: a tiny transformer used only to verify the framework mechanics."""

import torch
from torch import nn
import pytest


class ToyBlock(nn.Module):
    """Minimal decoder-style block: attention + MLP with residual connections."""

    def __init__(self, dim: int, num_heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class TupleBlock(nn.Module):
    """Block returning a tuple, Hugging Face style: (hidden_states, extra)."""

    def __init__(self, dim: int):
        super().__init__()
        self.inner = ToyBlock(dim)

    def forward(self, x):
        return (self.inner(x), None)


class ToyTransformer(nn.Module):
    """Tiny transformer language model for framework testing."""

    def __init__(self, vocab_size: int = 64, dim: int = 32, num_layers: int = 2, max_len: int = 32):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList(ToyBlock(dim) for _ in range(num_layers))
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        x = self.embed(ids) + self.pos(positions)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ToyTransformer()


@pytest.fixture
def sample_ids():
    return torch.randint(0, 64, (2, 8))
