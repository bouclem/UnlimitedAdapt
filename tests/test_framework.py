"""Tests for the adapter framework mechanics."""

import torch

from unlimitedadapt import (
    AdapterManager,
    MLPAdapter,
    freeze_base_model,
    adapter_parameters,
)
from unlimitedadapt.adapters import adapter_from_config
from unlimitedadapt.injection import inject_at, eject_at, AdaptedModule
from unlimitedadapt.training import count_trainable
from conftest import ToyBlock, TupleBlock


DIM = 32


def make_adapter(hidden=16):
    torch.manual_seed(42)
    return MLPAdapter(dim=DIM, hidden_dim=hidden)


def disturb(adapter):
    """Give an adapter non-zero output by randomizing its up-projection."""
    with torch.no_grad():
        adapter.up.weight.normal_(0, 0.1)
        adapter.up.bias.normal_(0, 0.1)


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


class TestTraining:
    def test_freeze_leaves_only_adapter_params_trainable(self, model):
        manager = AdapterManager(model)
        manager.attach("layers.0", "skill-a", make_adapter())
        freeze_base_model(model)
        trainable, total = count_trainable(model)
        adapter_count = sum(p.numel() for p in adapter_parameters(model))
        assert trainable == adapter_count
        assert trainable < total

    def test_adapter_trains_on_copy_task(self, model):
        """Smoke test: loss decreases when only the adapter is trained."""
        torch.manual_seed(0)
        manager = AdapterManager(model)
        manager.attach("layers.0", "skill-a", make_adapter())
        freeze_base_model(model)

        ids = torch.randint(0, 64, (8, 16))
        inputs, targets = ids[:, :-1], ids[:, 1:]
        optimizer = torch.optim.Adam(adapter_parameters(model), lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss()

        losses = []
        model.train()
        for _ in range(30):
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, 64), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]

    def test_base_weights_unchanged_by_training(self, model):
        manager = AdapterManager(model)
        manager.attach("layers.0", "skill-a", make_adapter())
        freeze_base_model(model)
        base_before = {
            name: p.clone()
            for name, p in model.named_parameters()
            if not p.requires_grad
        }

        ids = torch.randint(0, 64, (4, 16))
        optimizer = torch.optim.Adam(adapter_parameters(model), lr=1e-2)
        loss = model(ids[:, :-1]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, p in model.named_parameters():
            if name in base_before:
                assert torch.equal(p, base_before[name]), f"base param {name} changed"


class TestPersistence:
    def test_save_load_roundtrip(self, model, sample_ids, tmp_path):
        import copy

        # snapshot clean base weights before injection (injection renames keys)
        base_state = copy.deepcopy(model.state_dict())
        manager = AdapterManager(model)
        adapter = make_adapter()
        disturb(adapter)
        manager.attach("layers.0", "skill-a", adapter, alpha=0.7)
        expected = model(sample_ids)

        save_dir = manager.save("skill-a", tmp_path / "skill-a")

        fresh = type(model)()
        fresh.load_state_dict(base_state)
        manager2 = AdapterManager(fresh)
        name = manager2.load(save_dir, "layers.0")
        manager2.set_alpha(name, 0.7)
        actual = fresh(sample_ids)
        assert torch.allclose(expected, actual, atol=1e-6)

    def test_config_roundtrip(self):
        adapter = make_adapter(hidden=24)
        rebuilt = adapter_from_config(adapter.config())
        assert rebuilt.dim == DIM
        assert rebuilt.hidden_dim == 24


class TestErrors:
    def test_unknown_family_raises_unsupported_model(self):
        import pytest
        from unlimitedadapt import UnsupportedModelError
        from unlimitedadapt.architectures import get_architecture

        with pytest.raises(UnsupportedModelError):
            get_architecture("nonexistent-model")

    def test_unsupported_version_raises(self):
        import pytest
        from unlimitedadapt import UnsupportedVersionError
        from unlimitedadapt.architectures import get_architecture

        spec = get_architecture("minvoid")
        with pytest.raises(UnsupportedVersionError):
            spec.require_supported_version("1.6")
        spec.require_supported_version("1.1")  # must not raise

    def test_duplicate_attach_raises(self, model):
        import pytest
        from unlimitedadapt import AdapterAttachmentError

        manager = AdapterManager(model)
        manager.attach("layers.0", "skill-a", make_adapter())
        with pytest.raises(AdapterAttachmentError):
            manager.attach("layers.0", "skill-a", make_adapter())

    def test_missing_adapter_raises(self, model):
        import pytest
        from unlimitedadapt import AdapterNotFoundError

        manager = AdapterManager(model)
        with pytest.raises(AdapterNotFoundError):
            manager.activate("nonexistent")


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
