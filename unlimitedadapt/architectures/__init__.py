"""Architecture support for UnlimitedAdapt.

Each model family is supported by its own module in this package.
To add support for a new architecture, create a new file here that
subclasses ArchitectureSpec and registers an instance — nothing else
needs to change. All family modules are imported automatically.

No families are registered yet: concrete SLM support will be added once
the target models are chosen.
"""

import pkgutil
from importlib import import_module
from pathlib import Path

from unlimitedadapt.architectures.base import (
    ArchitectureSpec,
    register_architecture,
    get_architecture,
    supported_families,
)

__all__ = [
    "ArchitectureSpec",
    "register_architecture",
    "get_architecture",
    "supported_families",
]


def _import_family_modules() -> None:
    """Import every module in this package so families self-register."""
    package_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name != "base":
            import_module(f"unlimitedadapt.architectures.{module_info.name}")


_import_family_modules()
