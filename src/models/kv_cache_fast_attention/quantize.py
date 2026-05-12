"""
kv_cache_fast_attention/quantize.py
-------------------------------------
Re-exports the shared quantisation codec from kv_cache.quantize.

The INT8/INT4 encode/decode logic is identical in sticky_kv_logic_fast_attention.py
and sticky_kv_logic_cummulative.py (same @staticmethod bodies).
"""

from src.models.kv_cache.quantize import (   # noqa: F401
    quantize_k_per_window,
    quantize_v_per_window,
    dequantize_from_quant,
)

__all__ = ["quantize_k_per_window", "quantize_v_per_window", "dequantize_from_quant"]
