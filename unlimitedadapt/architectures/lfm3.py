"""Architecture support for LFM3 (LiquidAI/LFM3-*).

Hybrid architecture with custom modeling code — loading requires
trust_remote_code=True (modeling_lfm3.py ships with the weights).
Blocks alternate short-convolution, full-attention and sliding-attention
layers. Blocks live in model.layers and return plain tensors.

Heterogeneous components, same pattern as LFM2.5: only feed_forward
exists on every layer. Use operator_point() for the per-layer operator.

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


class Lfm3Spec(ArchitectureSpec):
    family = "lfm3"
    blocks_path = "model.layers"
    supported_versions = ""
    components = {"ffn": "feed_forward"}
    cache_style = "hybrid-kv-conv"
    cache_reset_on_adapter_change = True
    encoding = {
        "type": "bpe",
        "notes": "LiquidAI BPE tokenizer (tokenizer.json ships with weights)",
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


register_architecture(Lfm3Spec())
