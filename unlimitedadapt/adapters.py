"""Parallel adapter modules.

A parallel adapter computes a delta from the same input its host module
receives. The host output becomes:

    y = F_base(x) + alpha * F_adapter(x)

Adapters are zero-initialized at their output so that at creation the
delta is exactly zero and the base model is undisturbed (ControlNet-style).
Training then lets the adapter gradually earn its influence.
"""

import torch
from torch import nn


class ParallelAdapter(nn.Module):
    """Base class for parallel adapters.

    Subclasses implement `delta(x)` returning a tensor of the same shape
    as the host module's output. Input shape must match output shape of
    the host module (true for transformer decoder blocks).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.delta(x)

    def config(self) -> dict:
        """Serializable description of the adapter architecture."""
        raise NotImplementedError


class MLPAdapter(ParallelAdapter):
    """Bottleneck MLP parallel adapter: down-proj, activation, up-proj.

    The up-projection is zero-initialized so the adapter starts as a
    no-op on the host model.
    """

    def __init__(self, dim: int, hidden_dim: int, activation: str = "gelu"):
        super().__init__(dim)
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.down = nn.Linear(dim, hidden_dim)
        self.act = self._make_activation(activation)
        self.up = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if name not in activations:
            raise ValueError(f"unknown activation {name!r}, choose from {sorted(activations)}")
        return activations[name]()

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))

    def config(self) -> dict:
        return {
            "type": "mlp",
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation_name,
        }


def adapter_from_config(config: dict) -> ParallelAdapter:
    """Rebuild an adapter from its serialized config."""
    types = {"mlp": MLPAdapter}
    adapter_type = config.get("type")
    if adapter_type not in types:
        raise ValueError(f"unknown adapter type {adapter_type!r}, choose from {sorted(types)}")
    kwargs = {k: v for k, v in config.items() if k != "type"}
    return types[adapter_type](**kwargs)
