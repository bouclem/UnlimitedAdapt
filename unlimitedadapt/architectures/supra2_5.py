"""Architecture support for Supra2.5 (SupraLabs/Supra2.5-*).

Qwen3-architecture model, built into transformers — no custom code.
Blocks live in model.layers and return plain tensors. Same injection
layout as llama-style families, but a distinct generation: small hidden
size (640) and full_attention on all 12 layers.

Injection targets:
    whole block:  model.layers.N
    attention:    model.layers.N.self_attn   (tuple output, normed input)
    mlp:          model.layers.N.mlp         (tensor output, normed input)

Cache: standard KV cache. Switching the active adapter set mid-generation
requires restarting generation.
"""

from torch import nn

from unlimitedadapt.architectures.base import ArchitectureSpec, register_architecture


class Supra25Spec(ArchitectureSpec):
    family = "supra2_5"
    blocks_path = "model.layers"
    supported_versions = ""
    components = {"attn": "self_attn", "mlp": "mlp"}
    cache_style = "kv"
    cache_reset_on_adapter_change = True
    encoding = {
        "type": "bpe",
        "vocab_size": 16384,
        "notes": "qwen-style BPE with a reduced vocab, tie_word_embeddings=True",
    }

    def hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size


register_architecture(Supra25Spec())
