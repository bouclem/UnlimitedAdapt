"""Tests for alpha blending and manual routing behavior."""

import torch

from unlimitedadapt import AdapterManager
from conftest import DIM, make_adapter, disturb


class TestBlending:
    def test_alpha_zero_disables_contribution(self, model, sample_ids):
        before = model(sample_ids)
        manager = AdapterManager(model)
        adapter = make_adapter()
        disturb(adapter)
        manager.attach("layers.0", "skill-a", adapter, alpha=0.0)
        assert torch.equal(before, model(sample_ids))

    def test_alpha_scales_output_linearly(self, model):
        # blending is linear at the injection point; downstream nonlinear
        # layers mean it is not exactly linear at the final model output
        manager = AdapterManager(model)
        adapter = make_adapter()
        disturb(adapter)
        manager.attach("layers.0", "skill-a", adapter)
        x = torch.randn(2, 4, DIM)
        block = model.layers[0]
        manager.set_alpha("skill-a", 1.0)
        out1 = block(x)
        manager.set_alpha("skill-a", 2.0)
        out2 = block(x)
        manager.deactivate("skill-a")
        out0 = block(x)
        assert torch.allclose(out2 - out0, 2 * (out1 - out0), atol=1e-6)

    def test_two_adapters_blend_additively(self, model):
        manager = AdapterManager(model)
        a, b = make_adapter(), make_adapter()
        disturb(a)
        disturb(b)
        manager.attach("layers.0", "skill-a", a)
        manager.attach("layers.0", "skill-b", b)
        x = torch.randn(2, 4, DIM)
        block = model.layers[0]
        both = block(x)
        manager.deactivate("skill-b")
        only_a = block(x)
        manager.deactivate("skill-a")
        manager.activate("skill-b")
        only_b = block(x)
        manager.deactivate("skill-b")
        none = block(x)
        assert torch.allclose(both - none, (only_a - none) + (only_b - none), atol=1e-6)

    def test_manual_routing_toggles(self, model, sample_ids):
        manager = AdapterManager(model)
        adapter = make_adapter()
        disturb(adapter)
        manager.attach("layers.0", "skill-a", adapter)
        assert manager.active_adapters() == ["skill-a"]
        on = model(sample_ids)
        manager.deactivate("skill-a")
        assert manager.active_adapters() == []
        off = model(sample_ids)
        assert not torch.equal(on, off)
        manager.activate("skill-a")
        assert torch.equal(on, model(sample_ids))
