"""
sticky_kv_logic_fast_attention.py  —  backward-compatible shim
---------------------------------------------------------------
All logic has moved to src/models/kv_cache_fast_attention/:

    helpers.py    repeat_kv, _make_causal_mask, apply_rotary_pos_emb_single
                  (re-exported from kv_cache.helpers — identical logic)
    quantize.py   quantize_k_per_window, quantize_v_per_window, dequantize_from_quant
                  (re-exported from kv_cache.quantize — identical logic)
    allocator.py  update_k_win_and_local_num
                  (re-exported from kv_cache.allocator — identical logic)
    eviction.py   find_logical_window_span, gather_window_from_current_kv,
                  evict_from_window_scores,
                  create_mask_and_evict_from_kv_cache_prompt_stage
                  (re-exported from kv_cache.eviction — identical logic)
    ledger.py     get_ledger_data  (FA stub — returns {} with warning)
    cache.py      STICKYKVCache_LayerWise  (FA variant: no tracking blocks)

This file re-exports the public symbols so that any existing code using

    from src.models.sticky_kv_logic_fast_attention import STICKYKVCache_LayerWise

continues to work without modification.
"""

from src.models.kv_cache_fast_attention import (
    STICKYKVCache_LayerWise,
    repeat_kv,
    _make_causal_mask,
    apply_rotary_pos_emb_single,
)

__all__ = [
    "STICKYKVCache_LayerWise",
    "repeat_kv",
    "_make_causal_mask",
    "apply_rotary_pos_emb_single",
]
