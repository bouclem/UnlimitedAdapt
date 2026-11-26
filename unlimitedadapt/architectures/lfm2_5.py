"""Architecture support for LFM2.5 (LiquidAI/LFM2.5-*).

Hybrid architecture built into transformers (model_type 'lfm2'): blocks
alternate between short-convolution layers and full-attention layers.
Blocks live in model.layers and return plain tensors.

Heterogeneous components — attention layers have self_attn, conv layers
have conv instead. Only feed_forward exists on every layer, so it is the
sole uniform component. Use operator_point() to target the per-layer
operator (attention or conv) correctly.

Injection targets:
    whole block:      model.layers.N
    feed_forward:     model.layers.N.feed_forward   (every layer)
    operator (varies): model.layers.N.self_attn  on attention layers
                       model.layers.N.conv       on conv layers

Cache: hybrid — KV cache on attention layers, conv state cache on conv
layers. Switching the active adapter set mid-generation requires
restarting generation.
"""

from torch import nn

from unlimitedadapt.architectures.base import ArchitectureSpec, register_architecture


class Lfm25Spec(ArchitectureSpec):
    family = "lfm2_5"
    blocks_path = "model.layers"
    supported_versions = ""
    components = {"ffn": "feed_forward"}
    cache_style = "hybrid-kv-conv"
    cache_reset_on_adapter_change = True
    encoding = {
        "type": "bpe",
        "vocab_size": 65536,
        "notes": "LiquidAI BPE tokenizer",
    }

    def hidden_size(self, model: nn.Module) -> int:
        return model.config.hidden_size

    def operator_kind(self, model: nn.Module, index: int) -> str:
        """'attention' or 'conv' for the given layer index."""
        layer_types = model.config.layer_types
        return "conv" if layer_types[index] == "conv" else "attention"

    def operator_point(self, model: nn.Module, index: int) -> str:
        """Dotted path of the layer's operator module (self_attn or conv)."""
        attr = "conv" if self.operator_kind(model, index) == "conv" else "self_attn"
        return f"{self.block_point(index)}.{attr}"


register_architecture(Lfm25Spec())
