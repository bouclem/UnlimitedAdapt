"""Tests for adapter persistence: save/load roundtrips and configs."""

import torch

from unlimitedadapt import AdapterManager
from unlimitedadapt.adapters import adapter_from_config
from conftest import DIM, make_adapter, disturb


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
