"""UnlimitedAdapt: modular knowledge and capability adapters.

Parallel adapters grafted onto frozen pretrained models, with
alpha blending and manual routing. Model-agnostic by design.
"""

from unlimitedadapt.adapters import ParallelAdapter, MLPAdapter
from unlimitedadapt.errors import (
    UnlimitedAdaptError,
    UnsupportedModelError,
    UnsupportedVersionError,
    AdapterAttachmentError,
    AdapterNotFoundError,
)
from unlimitedadapt.injection import AdaptedModule, inject_at, eject_at
from unlimitedadapt.manager import AdapterManager
from unlimitedadapt.registry import AdapterInfo, save_adapter, load_adapter
from unlimitedadapt.training import freeze_base_model, adapter_parameters

__all__ = [
    "ParallelAdapter",
    "MLPAdapter",
    "AdaptedModule",
    "inject_at",
    "eject_at",
    "AdapterManager",
    "AdapterInfo",
    "save_adapter",
    "load_adapter",
    "freeze_base_model",
    "adapter_parameters",
    "UnlimitedAdaptError",
    "UnsupportedModelError",
    "UnsupportedVersionError",
    "AdapterAttachmentError",
    "AdapterNotFoundError",
]

__version__ = "0.1.1"
