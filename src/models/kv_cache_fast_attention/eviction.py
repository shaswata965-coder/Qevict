"""
kv_cache_fast_attention/eviction.py
-------------------------------------
Re-exports the shared eviction helpers from kv_cache.eviction.

The four eviction / span-lookup functions have identical logic in both
sticky_kv_logic_fast_attention.py and sticky_kv_logic_cummulative.py.
The fast-attention path omits per-token tracking but that tracking happens
inside __call__, not inside the eviction functions themselves.
"""

from src.models.kv_cache.eviction import (   # noqa: F401
    find_logical_window_span,
    gather_window_from_current_kv,
    evict_from_window_scores,
    create_mask_and_evict_from_kv_cache_prompt_stage,
)

__all__ = [
    "find_logical_window_span",
    "gather_window_from_current_kv",
    "evict_from_window_scores",
    "create_mask_and_evict_from_kv_cache_prompt_stage",
]
