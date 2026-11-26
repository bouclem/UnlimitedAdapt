"""Architecture support for MinVoid (Sqersters/MinVoid-*), versions <= 1.1.

Structure (1.0/1.1): byte-level GPT with GLA attention, sequential blocks
(separate n2 norm before the MLP). Blocks live in model.blocks and return
(hidden_states, latent_cache) tuples, which AdaptedModule handles natively.

Injection targets:
    whole block:  model.blocks.N          input = block input (residual stream)
    attention:    model.blocks.N.attn     input = normed input, tuple output
    mlp:          model.blocks.N.mlp      input = normed input, tensor output

Cache: per-layer latent tensors. Switching the active adapter set
mid-generation makes the cache inconsistent (it was computed under the
old configuration), so generation must restart from scratch after a
routing change.

Versions 1.6+ restructured the block (parallel attention/MLP, qk-norm) —
those belong in a separate structural spec and are not supported here.
"""

from torch import nn

from unlimitedadapt.architectures.base import ArchitectureSpec, register_architecture


class MinVoidSpec(ArchitectureSpec):
    family = "minvoid"
    blocks_path = "model.blocks"
    supported_versions = "<=1.1"
    components = {"attn": "attn", "mlp": "mlp"}
    cache_style = "latent"
    cache_reset_on_adapter_change = True
    encoding = {
        "type": "bpe",
        "vocab_size": 258,
        "bos": 256,
        "eos": 257,
        "notes": "BPE tokenizer (tokenizer.json) over a byte-level alphabet",
    }

    def hidden_size(self, model: nn.Module) -> int:
        return model.config.d_model


register_architecture(MinVoidSpec())
