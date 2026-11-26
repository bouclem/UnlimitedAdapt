"""Tests for framework error types and their raising conditions."""

import pytest

from unlimitedadapt import (
    AdapterManager,
    UnsupportedModelError,
    UnsupportedVersionError,
    AdapterAttachmentError,
    AdapterNotFoundError,
)
from unlimitedadapt.architectures import get_architecture
from conftest import make_adapter


class TestErrors:
    def test_unknown_family_raises_unsupported_model(self):
        with pytest.raises(UnsupportedModelError):
            get_architecture("nonexistent-model")

    def test_unsupported_version_raises(self):
        spec = get_architecture("minvoid")
        with pytest.raises(UnsupportedVersionError):
            spec.require_supported_version("1.6")
        spec.require_supported_version("1.1")  # must not raise

    def test_duplicate_attach_raises(self, model):
        manager = AdapterManager(model)
        manager.attach("layers.0", "skill-a", make_adapter())
        with pytest.raises(AdapterAttachmentError):
            manager.attach("layers.0", "skill-a", make_adapter())

    def test_missing_adapter_raises(self, model):
        manager = AdapterManager(model)
        with pytest.raises(AdapterNotFoundError):
            manager.activate("nonexistent")
