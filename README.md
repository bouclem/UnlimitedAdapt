# UnlimitedAdapt

Give your language model new skills without retraining it.

UnlimitedAdapt lets you attach small, independent **adapters** to a frozen
pretrained model. Each adapter can be trained on its own, then turned on,
turned off, or blended at partial strength — whenever you want, even
mid-conversation. The base model is never modified.

## How it works

Every adapter is a tiny module that sits next to part of the model and
nudges its output:

```
output = model(x) + strength * adapter(x)
```

- Several adapters can be active at once — their effects add up
- New adapters start completely silent, so attaching one never disturbs
  the model
- The `strength` dial (alpha) means a skill doesn't have to be all-or-nothing

## Install

Not on PyPI — install straight from this folder:

```
pip install -e .
```

The `-e` flag means any changes you make to the code apply right away,
no reinstall needed. If you want to run the tests too: `pip install -e .[dev]`

## Quick example

```python
from unlimitedadapt import AdapterManager, MLPAdapter, freeze_base_model

manager = AdapterManager(model)
manager.attach("layers.0", "python-performance", MLPAdapter(dim=512, hidden_dim=128))

# you choose which skills are active
manager.deactivate("python-performance")
manager.activate("python-performance")
manager.set_alpha("python-performance", 0.7)   # use it at 70% strength

# training: the base model stays frozen, only the adapter learns
freeze_base_model(model)

# save it, load it back later
manager.save("python-performance", "adapters/python-performance.v1")
manager.load("adapters/python-performance.v1", point="layers.0")
```

## Good to know

- **Load the model's weights first, then attach adapters** — the other
  way around confuses checkpoint loading
- Strength blending is exact where the adapter sits, but only approximate
  in the model's final output (the rest of the model reshapes it a little)
- Right now each adapter attaches to one spot in the model

## Supporting a specific model

Each model family gets its own small plug-in file under
`unlimitedadapt/architectures/`. None exist yet — they'll be added once
the target models are chosen.

## Tests

```
python -m pytest tests/ -q
```

## Roadmap

- [ ] Support for real SLMs (models to be chosen)
- [ ] Small-scale experiment proving adapters beat a same-sized plain model
- [ ] Adapters that span several layers at once
- [ ] Automatic skill selection based on what you ask
- [ ] Exotic adapter types (experimental)
