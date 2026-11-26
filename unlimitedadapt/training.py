"""Training utilities for adapters.

Adapters are trained with the base model frozen: gradients flow through
the base (needed for backprop into the adapter) but only adapter
parameters are updated by the optimizer.
"""

from torch import nn

from unlimitedadapt.injection import AdaptedModule


def freeze_base_model(model: nn.Module) -> int:
    """Freeze every parameter that is not inside an adapter.

    Returns the number of frozen parameters.
    """
    adapter_param_ids = {id(p) for p in adapter_parameters(model)}
    frozen = 0
    for param in model.parameters():
        if id(param) in adapter_param_ids:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def adapter_parameters(model: nn.Module) -> list:
    """All parameters belonging to adapters attached anywhere in the model."""
    params = []
    for module in model.modules():
        if isinstance(module, AdaptedModule):
            for adapter in module.adapters.values():
                params.extend(adapter.parameters())
    return params


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(trainable params, total params) — useful for experiment logs."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
