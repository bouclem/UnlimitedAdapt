"""Architecture support for SmolLM2 (HuggingFaceTB/SmolLM2-*).

Standard llama-architecture model, built into transformers — no custom
code needed. Blocks live in model.layers and return plain tensors
(transformers 5.x style, no tuple).

Injection targets:
    whole block:  model.layers.N
    attention:    model.layers.N.self_attn   (tuple output, normed input)
    mlp:          model.layers.N.mlp         (tensor output, normed input)

Cache: standard KV cache. Switching the active adapter set mid-generation
requires restarting generation (cache computed under old configuration).
"""

from torch import nn

from unlimitedadapt.architectures.base import ArchitectureSpec, register_architecture


class SmolLM2Spec(ArchitectureSpec):
    family = "smollm2"
    blocks_path = "model.layers"
    supported_versions = ""  # all sizes (135M, 360M, 1.7B) share this structure
    components = {"attn": "self_attn", "mlp": "mlp"}
    cache_style = "kv"
    cache_reset_on_adapter_change = True
    encoding = {
        "type": "bpe",
        "vocab_size": 49152,
        "notes": "llama-style BPE, tie_word_embeddings=True on small sizes",
    }

    def hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size


register_architecture(SmolLM2Spec())
