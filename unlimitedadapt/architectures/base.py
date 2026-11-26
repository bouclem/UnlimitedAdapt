"""Base class and registry for architecture support.

Each supported model family lives in its own module inside this package
(e.g. architectures/minvoid.py). A family module defines a subclass of
ArchitectureSpec and registers an instance at import time.

Naming rule: one spec per STRUCTURAL generation. If two versions of a
model differ structurally (different block layout, different attention),
they get separate specs with precise names — never one vague family
lumping them together. supported_versions is only for minor versions
sharing the same structure.
"""

import re

from torch import nn

from unlimitedadapt.errors import UnsupportedModelError, UnsupportedVersionError
from unlimitedadapt.injection import get_module_at


def parse_version(tag: str) -> tuple[int, ...]:
    """Turn '1.6' into (1, 6) for comparison."""
    return tuple(int(part) for part in tag.split("."))


def version_matches(expr: str, version: str) -> bool:
    """Check a version against a range expression.

    Supported forms: '<=1.1', '>=1.0', '<2.0', '>1.5', '1.6' (exact),
    and comma-separated combinations like '>=1.0,<1.6'.
    """
    target = parse_version(version)
    for clause in expr.split(","):
        clause = clause.strip()
        match = re.fullmatch(r"(<=|>=|<|>|==)?\s*(\d+(?:\.\d+)*)", clause)
        if not match:
            raise ValueError(f"unparseable version expression {expr!r}")
        op, bound_tag = match.groups()
        bound = parse_version(bound_tag)
        if op == "<=" and not target <= bound:
            return False
        if op == ">=" and not target >= bound:
            return False
        if op == "<" and not target < bound:
            return False
        if op == ">" and not target > bound:
            return False
        if op in (None, "==") and not target == bound:
            return False
    return True


class ArchitectureSpec:
    """Describes how adapters integrate with one structural model generation.

    family: precise name of the structural generation, e.g. 'minvoid'
        or 'qwen3' — never a vague umbrella over different structures.
    blocks_path: dotted path to the module list holding the repeatable
        blocks, e.g. 'model.blocks' or 'model.layers'.
    supported_versions: range expression, e.g. '<=1.1' or '>=1.0,<1.6'.
        Only minor versions sharing this exact structure.
    components: injectable submodules inside each block, as
        label -> attribute name, e.g. {'attn': 'attn', 'mlp': 'mlp'}.
        Whole-block injection is always available; components give
        finer-grained targets.
    cache_style: how generation caching works — 'kv', 'latent', 'none',
        or a family-specific tag.
    cache_reset_on_adapter_change: whether the generation cache must be
        discarded when the active adapter set changes mid-conversation.
        True for essentially every current architecture: cached state was
        computed with the old adapter configuration, so continuing to
        use it after a switch mixes inconsistent computations.
    encoding: how text becomes tokens, e.g.
        {'type': 'byte-level', 'bos': 256, 'eos': 257}.
    """

    family: str = ""
    blocks_path: str = ""
    supported_versions: str = ""
    components: dict[str, str] = {}
    cache_style: str = ""
    cache_reset_on_adapter_change: bool = True
    encoding: dict = {}

    # --- versions ---

    def is_version_supported(self, version: str) -> bool:
        if not self.supported_versions:
            return True
        return version_matches(self.supported_versions, version)

    def require_supported_version(self, version: str) -> None:
        """Raise UnsupportedVersionError if the version is not supported."""
        if not self.is_version_supported(version):
            raise UnsupportedVersionError(
                f"{self.family} {version} is not supported; "
                f"supported versions: {self.supported_versions or 'all'}"
            )

    # --- injection layout ---

    def num_layers(self, model: nn.Module) -> int:
        return len(get_module_at(model, self.blocks_path))

    def hidden_size(self, model: nn.Module) -> int:
        """Hidden dimension of the blocks, needed to size adapters."""
        raise NotImplementedError(
            f"architecture {self.family!r} must implement hidden_size()"
        )

    def block_point(self, index: int) -> str:
        """Dotted path of one whole block, usable as an injection point."""
        return f"{self.blocks_path}.{index}"

    def block_points(self, model: nn.Module) -> list[str]:
        """Dotted paths of every injectable block in the model."""
        return [self.block_point(i) for i in range(self.num_layers(model))]

    def component_point(self, index: int, component: str) -> str:
        """Dotted path of a finer injection target inside one block.

        e.g. component_point(0, 'attn') -> 'model.blocks.0.attn'
        """
        if component not in self.components:
            raise KeyError(
                f"{self.family} blocks have no component {component!r}; "
                f"available: {sorted(self.components) or '(none)'}"
            )
        return f"{self.block_point(index)}.{self.components[component]}"

    def component_points(self, model: nn.Module) -> list[str]:
        """Dotted paths of every component injection target in the model."""
        return [
            self.component_point(i, component)
            for i in range(self.num_layers(model))
            for component in self.components
        ]

    def injection_points(self, model: nn.Module) -> dict[str, list[str]]:
        """All injection targets, grouped: whole blocks and components."""
        return {
            "blocks": self.block_points(model),
            "components": self.component_points(model),
        }


_REGISTRY: dict[str, ArchitectureSpec] = {}


def register_architecture(spec: ArchitectureSpec) -> None:
    if not spec.family:
        raise ValueError("architecture spec must define a family name")
    _REGISTRY[spec.family] = spec


def get_architecture(family: str) -> ArchitectureSpec:
    if family not in _REGISTRY:
        raise UnsupportedModelError(
            f"architecture {family!r} is not supported yet; "
            f"registered families: {sorted(_REGISTRY) or '(none)'}. "
            f"To add support, create a spec module in unlimitedadapt/architectures/."
        )
    return _REGISTRY[family]


def supported_families() -> list[str]:
    return sorted(_REGISTRY)
