"""Injection machinery: graft adapters onto any nn.Module tree.

The injection layer is model-agnostic. Any module in a model can be
wrapped in an AdaptedModule, which holds parallel adapters and blends
their output into the base module's output:

    y = base(x) + sum(alpha_i * adapter_i(x) for active adapters)

The adapter receives the same input x as the base module (parallel
design). Works with modules returning either a plain tensor or a tuple
whose first element is the main hidden state (Hugging Face style).
"""

import torch
from torch import nn

from unlimitedadapt.adapters import ParallelAdapter
from unlimitedadapt.errors import AdapterAttachmentError, AdapterNotFoundError


class AdaptedModule(nn.Module):
    """Wraps a base module and adds parallel adapter outputs to it."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self._alphas: dict[str, float] = {}
        self._active: dict[str, bool] = {}

    def attach(self, name: str, adapter: ParallelAdapter, alpha: float = 1.0) -> None:
        if name in self.adapters:
            raise AdapterAttachmentError(f"adapter {name!r} already attached at this point")
        # match the host module's dtype and device — adapters are created
        # in fp32/CPU by default but pretrained models are often bf16/GPU
        try:
            reference = next(self.base.parameters())
            adapter = adapter.to(dtype=reference.dtype, device=reference.device)
        except StopIteration:
            pass
        self.adapters[name] = adapter
        self._alphas[name] = alpha
        self._active[name] = True

    def detach(self, name: str) -> ParallelAdapter:
        if name not in self.adapters:
            raise AdapterNotFoundError(f"no adapter {name!r} attached at this point")
        adapter = self.adapters.pop(name)
        del self._alphas[name]
        del self._active[name]
        return adapter

    def set_active(self, name: str, active: bool) -> None:
        self._require(name)
        self._active[name] = active

    def set_alpha(self, name: str, alpha: float) -> None:
        self._require(name)
        self._alphas[name] = alpha

    def alpha(self, name: str) -> float:
        self._require(name)
        return self._alphas[name]

    def is_active(self, name: str) -> bool:
        self._require(name)
        return self._active[name]

    def adapter_names(self) -> list[str]:
        return list(self.adapters.keys())

    def _require(self, name: str) -> None:
        if name not in self.adapters:
            raise AdapterNotFoundError(f"no adapter {name!r} attached at this point")

    def blended_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of alpha-scaled outputs of all active adapters."""
        delta = None
        for name, adapter in self.adapters.items():
            if not self._active[name]:
                continue
            contribution = self._alphas[name] * adapter(x)
            delta = contribution if delta is None else delta + contribution
        return delta

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out = self.base(x, *args, **kwargs)
        if isinstance(out, tuple):
            main, rest = out[0], out[1:]
        else:
            main, rest = out, None

        delta = self.blended_delta(x)
        if delta is not None:
            main = main + delta

        if rest is not None:
            return (main, *rest)
        return main


def get_module_at(model: nn.Module, path: str) -> nn.Module:
    """Fetch a submodule by dotted path, e.g. 'model.layers.3'."""
    module = model
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_at(model: nn.Module, path: str, replacement: nn.Module) -> None:
    """Replace a submodule by dotted path."""
    parts = path.split(".")
    parent = get_module_at(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = replacement
    else:
        setattr(parent, last, replacement)


def inject_at(model: nn.Module, path: str) -> AdaptedModule:
    """Wrap the module at `path` in an AdaptedModule, in place.

    Returns the wrapper so adapters can be attached to it.
    """
    target = get_module_at(model, path)
    if isinstance(target, AdaptedModule):
        return target
    wrapper = AdaptedModule(target)
    set_module_at(model, path, wrapper)
    return wrapper


def eject_at(model: nn.Module, path: str) -> nn.Module:
    """Remove the AdaptedModule wrapper at `path`, restoring the base module."""
    target = get_module_at(model, path)
    if not isinstance(target, AdaptedModule):
        raise ValueError(f"module at {path!r} is not adapted")
    set_module_at(model, path, target.base)
    return target.base
