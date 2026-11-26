"""Architecture support for GPT-2 (openai-community/gpt2 and finetunes).

Standard HuggingFace GPT-2, built into transformers. Blocks live in
transformer.h and return plain tensors (transformers 5.x style).
All sizes (117M/345M/774M/1.5B) share this structure.

Injection targets:
    whole block:  transformer.h.N
    attention:    transformer.h.N.attn   (tuple output, normed input)
    mlp:          transformer.h.N.mlp    (tensor output, normed input)

Note: GPT2Attention uses Conv1D rather than Linear internally — irrelevant
for parallel adapters, which only see inputs and outputs.

Cache: standard KV cache. Switching the active adapter set mid-generation
requires restarting generation.
"""

from torch import nn

from unlimitedadapt.architectures.base import ArchitectureSpec, register_architecture


class Gpt2Spec(ArchitectureSpec):
    family = "gpt2"
    blocks_path = "transformer.h"
    supported_versions = ""
    components = {"attn": "attn", "mlp": "mlp"}
    cache_style = "kv"
    cache_reset_on_adapter_change = True
    encoding = {
        "type": "bpe",
        "vocab_size": 50257,
        "bos": 50256,
        "eos": 50256,
        "notes": "original GPT-2 BPE; <|endoftext|> serves as both bos and eos",
    }

    def hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size


register_architecture(Gpt2Spec())
