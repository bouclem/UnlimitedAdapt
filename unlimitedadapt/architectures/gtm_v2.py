"""Architecture support for GTM-v2 (OPENGCM/GTM-v2-*).

nanoGPT-style plain PyTorch model — NOT a HuggingFace transformers model.
The released weights load manually: instantiate GPT/GPTConfig from the
repo's model.py and load model.safetensors into it (see the repo README).
Note: config.json contains extra keys GPTConfig rejects (e.g. model_type)
— filter to dataclass fields before instantiating.

Structure: classic GPT-2-style blocks at transformer.h, plain tensor in,
plain tensor out, no cache implementation (the reference generate() is
uncached).

Injection targets:
    whole block:  transformer.h.N
    attention:    transformer.h.N.attn    (tensor output, normed input)
    mlp:          transformer.h.N.mlp     (tensor output, normed input)

Note: the model object has no .config.hidden_size — it is a dataclass
config, so hidden size comes from config.n_embd.
"""

from torch import nn

from unlimitedadapt.architectures.base import ArchitectureSpec, register_architecture


class GtmV2Spec(ArchitectureSpec):
    family = "gtm-v2"
    blocks_path = "transformer.h"
    supported_versions = ""
    components = {"attn": "attn", "mlp": "mlp"}
    cache_style = "none"
    cache_reset_on_adapter_change = False  # no cache exists to invalidate
    encoding = {
        "type": "bpe",
        "vocab_size": 50257,
        "bos": 50256,
        "eos": 50256,
        "notes": "tiktoken gpt2 encoding; <|endoftext|> serves as both bos and eos",
    }

    def hidden_size(self, model: nn.Module) -> int:
        return model.config.n_embd


register_architecture(GtmV2Spec())
