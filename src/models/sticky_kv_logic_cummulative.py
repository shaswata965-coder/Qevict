"""
sticky_kv_logic_cummulative.py  —  backward-compatible shim
------------------------------------------------------------
All logic has moved to src/models/kv_cache/:

    helpers.py    repeat_kv, _make_causal_mask, apply_rotary_pos_emb_single
    quantize.py   quantize_k_per_window, quantize_v_per_window, dequantize_from_quant
    allocator.py  update_k_win_and_local_num  (CFO budget allocator)
    eviction.py   find_logical_window_span, gather_window_from_current_kv,
                  evict_from_window_scores,
                  create_mask_and_evict_from_kv_cache_prompt_stage
    ledger.py     get_ledger_data
    cache.py      STICKYKVCache_LayerWise  (thin orchestrator)

This file re-exports the public symbols so that any existing code using

    from src.models.sticky_kv_logic_cummulative import STICKYKVCache_LayerWise

continues to work without modification.
"""

from src.models.kv_cache import (
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