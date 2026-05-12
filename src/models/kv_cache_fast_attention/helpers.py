"""
kv_cache_fast_attention/helpers.py
-----------------------------------
Re-exports the shared tensor helpers from kv_cache.helpers.

The fast-attention variant uses the same repeat_kv, _make_causal_mask, and
apply_rotary_pos_emb_single as the cumulative variant — their logic is
identical in both source files.
"""

from src.models.kv_cache.helpers import (   # noqa: F401
    repeat_kv,
    _make_causal_mask,
    apply_rotary_pos_emb_single,
)

__all__ = ["repeat_kv", "_make_causal_mask", "apply_rotary_pos_emb_single"]
