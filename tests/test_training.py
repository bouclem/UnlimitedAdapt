"""Tests for training behavior: freezing, adapter-only updates, base integrity."""

import torch

from unlimitedadapt import AdapterManager, freeze_base_model, adapter_parameters
from unlimitedadapt.training import count_trainable
from conftest import make_adapter


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
