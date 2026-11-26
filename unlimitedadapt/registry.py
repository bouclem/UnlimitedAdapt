"""Adapter registry: metadata and persistence.

Each adapter is stored as a directory:

    my_adapter/
        adapter.json    metadata: name, version, description, architecture
        weights.pt      state dict

The metadata doubles as documentation during manual routing and later
as retrieval input for automatic routing.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from unlimitedadapt.adapters import ParallelAdapter, adapter_from_config

METADATA_FILE = "adapter.json"
WEIGHTS_FILE = "weights.pt"


@dataclass
class AdapterInfo:
    """Metadata describing an adapter."""

    name: str
    version: str = "1"
    description: str = ""
    architecture: dict = field(default_factory=dict)
    default_alpha: float = 1.0
    example_queries: list[str] = field(default_factory=list)

    def full_name(self) -> str:
        return f"{self.name}.v{self.version}" if self.version else self.name


def save_adapter(adapter: ParallelAdapter, info: AdapterInfo, directory: str | Path) -> Path:
    """Save adapter weights and metadata to a directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    info.architecture = adapter.config()
    with open(directory / METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(info), f, indent=2)

    torch.save(adapter.state_dict(), directory / WEIGHTS_FILE)
    return directory


def load_adapter(directory: str | Path) -> tuple[ParallelAdapter, AdapterInfo]:
    """Load an adapter and its metadata from a directory."""
    directory = Path(directory)
    with open(directory / METADATA_FILE, encoding="utf-8") as f:
        info = AdapterInfo(**json.load(f))

    adapter = adapter_from_config(info.architecture)
    state = torch.load(directory / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state)
    return adapter, info
