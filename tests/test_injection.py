"""Tests for injection mechanics: wrapping, ejecting, output formats,
and dtype/device matching."""

import torch

from unlimitedadapt import AdapterManager
from unlimitedadapt.injection import inject_at, eject_at, AdaptedModule
from conftest import DIM, TupleBlock, make_adapter, disturb


class TestInjection:
    def test_injected_model_output_unchanged_at_init(self, model, sample_ids):
        before = model(sample_ids)
        manager = AdapterManager(model)
        manager.attach("layers.0", "skill-a", make_adapter())
        after = model(sample_ids)
        assert torch.equal(before, after)

    def test_eject_restores_base_module(self, model):
        original_block = model.layers[0]
        inject_at(model, "layers.0")
        assert isinstance(model.layers[0], AdaptedModule)
        eject_at(model, "layers.0")
        assert model.layers[0] is original_block

    def test_tuple_returning_block_supported(self):
        torch.manual_seed(0)
        block = TupleBlock(DIM)
        wrapper = AdaptedModule(block)
        wrapper.attach("a", make_adapter())
        x = torch.randn(2, 4, DIM)
        out = wrapper(x)
        assert isinstance(out, tuple)
        assert out[0].shape == x.shape


class TestDtypeDeviceMatching:
    def test_adapter_matches_base_dtype(self, model, sample_ids):
        """Real pretrained models are often bf16 — adapters must follow."""
        model = model.to(torch.bfloat16)
        manager = AdapterManager(model)
        manager.attach("layers.0", "skill-a", make_adapter())  # fp32 by default
        adapter = model.layers[0].adapters["skill-a"]
        assert next(adapter.parameters()).dtype == torch.bfloat16
        out = model(sample_ids)  # must not raise a dtype error
        assert out.dtype == torch.bfloat16
