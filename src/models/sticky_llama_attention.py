"""
sticky_llama_attention.py  —  backward-compatible shim
------------------------------------------------------
All logic has moved to src/models/attention/:

    rope.py        Llama3RotaryEmbedding, init_rope()
    ops.py         compute_main_logits, apply_prefill_causal_mask,
                   compute_qcache_joint_softmax, compute_standard_softmax
    module.py      STICKYLlamaAttention  (standard SDPA backend)

This file re-exports the public symbols so that any existing code using

    from src.models.sticky_llama_attention import STICKYLlamaAttention

continues to work without modification.
"""

from src.models.attention.module import STICKYLlamaAttention
from src.models.attention.rope import Llama3RotaryEmbedding, init_rope
from src.models.sticky_kv_logic_cummulative import (
    repeat_kv,
    _make_causal_mask,
    apply_rotary_pos_emb_single,
    STICKYKVCache_LayerWise,
)

__all__ = [
    "STICKYLlamaAttention",
    "Llama3RotaryEmbedding",
    "init_rope",
    # Legacy KV-logic helpers kept for any callers that imported them transitively
    "repeat_kv",
    "_make_causal_mask",
    "apply_rotary_pos_emb_single",
    "STICKYKVCache_LayerWise",
]