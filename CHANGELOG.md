# Changelog

## [0.1.1] - 2026-11-26

### Added

- First model family support: **MinVoid** versions <= 1.1
  (`architectures/minvoid.py`) — verified live against Sqersters/MinVoid-1.1:
  adapters attach silently, routing toggles cleanly
- Architecture specs now cover more than paths: supported version ranges
  (`<=1.1` style), finer injection points (attention/MLP components inside
  blocks, not just whole blocks), cache behavior for generation, and
  tokenizer/encoding info
- Naming rule for specs: one spec per structural generation — structurally
  different versions (like MinVoid 1.6) get their own spec, never lumped
  under one family name
- Framework errors (`unlimitedadapt/errors.py`): unsupported model,
  unsupported version, attachment conflicts, missing adapters — each with
  its own exception type and an actionable message

## [0.1.0] - 2026-11-26

First version — the core framework is up and running.

### What's in

- Parallel adapters you can attach to any model, starting silent so the
  model is never disturbed
- Turn adapters on and off, or dial them up and down with a strength knob
- Several adapters can work at the same time — their effects add up
- Save and load trained adapters, each with its own info card (name,
  description, what it's good at)
- Training setup that freezes the base model so only adapters learn
- A plug-in system for supporting different model families (none added yet)
- A test suite checking all of the above — 12 tests, all passing
