"""
sticky_llama_attention_fast_attention.py  —  backward-compatible shim
----------------------------------------------------------------------
All logic has moved to src/models/attention/:

    rope.py          Llama3RotaryEmbedding, init_rope()
    ops.py           compute_main_logits, compute_qcache_joint_softmax,
                     compute_standard_softmax   (shared with SDPA backend)
    ops_flash.py     prefill_flash_attention, compute_chunked_prefill_scores
    module_flash.py  STICKYLlamaAttention  (Flash-Attention v2 backend)

This file re-exports the public symbols so that any existing code using

    from sticky_llama_attention_fast_attention import STICKYLlamaAttention

continues to work without modification.
"""

from src.models.attention.module_flash import STICKYLlamaAttention
from src.models.attention.rope import Llama3RotaryEmbedding, init_rope
from src.models.sticky_kv_logic_fast_attention import (
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