"""
kv_cache_fast_attention/__init__.py
-------------------------------------
Public API for the kv_cache_fast_attention sub-package.

Re-exports every symbol that was previously importable from
``src.models.sticky_kv_logic_fast_attention`` so all existing call sites
continue to work with zero modifications.

Typical usage (unchanged from before):

    from src.models.kv_cache_fast_attention import STICKYKVCache_LayerWise
    from src.models.kv_cache_fast_attention import repeat_kv, _make_causal_mask

The flash-attention backend (attention/ops_flash.py) calls
``kv_cache._dequantize_from_quant(...)``; that @staticmethod wrapper is kept
on the class inside cache.py, so nothing in ops_flash.py needs to change.
"""

from .cache import STICKYKVCache_LayerWise
from .helpers import repeat_kv, _make_causal_mask, apply_rotary_pos_emb_single

__all__ = [
    "STICKYKVCache_LayerWise",
    "repeat_kv",
    "_make_causal_mask",
    "apply_rotary_pos_emb_single",
]
