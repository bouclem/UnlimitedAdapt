"""AdapterManager: the layer between the model and its adapters.

Handles injection points on the model, adapter attachment, and manual
routing (which adapters are active, at what alpha). This is the phase-1
routing interface: the user explicitly selects adapters, e.g.:

    manager.activate("python-performance")
    manager.set_alpha("python-performance", 0.8)

Automatic routing can later be built on top by driving the same API.
"""

from pathlib import Path

from torch import nn

from unlimitedadapt.adapters import ParallelAdapter
from unlimitedadapt.errors import AdapterAttachmentError, AdapterNotFoundError
from unlimitedadapt.injection import AdaptedModule, inject_at, eject_at
from unlimitedadapt.registry import AdapterInfo, save_adapter, load_adapter


class AdapterManager:
    """Manages adapters grafted onto a model."""

    def __init__(self, model: nn.Module):
        self.model = model
        # adapter name -> (injection point path, adapter info)
        self._attachments: dict[str, tuple[str, AdapterInfo]] = {}

    def inject(self, point: str) -> AdaptedModule:
        """Prepare an injection point on the model, e.g. 'model.layers.3'."""
        return inject_at(self.model, point)

    def eject(self, point: str) -> None:
        """Remove an injection point, detaching any adapters on it."""
        wrapper = self._wrapper(point)
        for name in wrapper.adapter_names():
            full = self._find_full_name(point, name)
            if full is not None:
                del self._attachments[full]
        eject_at(self.model, point)

    def attach(
        self,
        point: str,
        name: str,
        adapter: ParallelAdapter,
        info: AdapterInfo | None = None,
        alpha: float | None = None,
    ) -> None:
        """Attach an adapter at an injection point and activate it."""
        if name in self._attachments:
            raise AdapterAttachmentError(f"adapter {name!r} is already attached")
        wrapper = self.inject(point)
        if info is None:
            info = AdapterInfo(name=name, architecture=adapter.config())
        if alpha is None:
            alpha = info.default_alpha
        wrapper.attach(name, adapter, alpha=alpha)
        self._attachments[name] = (point, info)

    def detach(self, name: str) -> ParallelAdapter:
        """Detach an adapter from the model entirely."""
        point, _ = self._require(name)
        wrapper = self._wrapper(point)
        adapter = wrapper.detach(name)
        del self._attachments[name]
        return adapter

    # --- manual routing ---

    def activate(self, name: str) -> None:
        point, _ = self._require(name)
        self._wrapper(point).set_active(name, True)

    def deactivate(self, name: str) -> None:
        point, _ = self._require(name)
        self._wrapper(point).set_active(name, False)

    def set_alpha(self, name: str, alpha: float) -> None:
        point, _ = self._require(name)
        self._wrapper(point).set_alpha(name, alpha)

    def active_adapters(self) -> list[str]:
        """Names of all attached adapters currently contributing to output."""
        result = []
        for name, (point, _) in self._attachments.items():
            if self._wrapper(point).is_active(name):
                result.append(name)
        return result

    def attached_adapters(self) -> list[str]:
        return list(self._attachments.keys())

    def info(self, name: str) -> AdapterInfo:
        _, info = self._require(name)
        return info

    # --- persistence ---

    def save(self, name: str, directory: str | Path) -> Path:
        """Save one adapter's trained weights and metadata."""
        point, info = self._require(name)
        wrapper = self._wrapper(point)
        return save_adapter(wrapper.adapters[name], info, directory)

    def load(self, directory: str | Path, point: str) -> str:
        """Load a saved adapter, attach it at `point`, and activate it.

        Returns the adapter name.
        """
        adapter, info = load_adapter(directory)
        self.attach(point, info.name, adapter, info=info)
        return info.name

    # --- internals ---

    def _require(self, name: str) -> tuple[str, AdapterInfo]:
        if name not in self._attachments:
            raise AdapterNotFoundError(f"adapter {name!r} is not attached")
        return self._attachments[name]

    def _wrapper(self, point: str) -> AdaptedModule:
        from unlimitedadapt.injection import get_module_at

        module = get_module_at(self.model, point)
        if not isinstance(module, AdaptedModule):
            raise AdapterAttachmentError(f"no injection point at {point!r}")
        return module

    def _find_full_name(self, point: str, name: str) -> str | None:
        if name in self._attachments and self._attachments[name][0] == point:
            return name
        return None
