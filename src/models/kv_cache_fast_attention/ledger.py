"""
kv_cache_fast_attention/ledger.py
-----------------------------------
Ledger accessor stub for the fast-attention KV-cache variant.

Extracted verbatim from STICKYKVCache_LayerWise.get_ledger_data()
(sticky_kv_logic_fast_attention.py, lines 1002–1015).

The fast-attention path omits per-token ledger tracking to avoid the
O(N²) memory overhead of the full prefill attention matrix.  Use the
cumulative module (kv_cache/) for research analysis.
"""

import warnings


def get_ledger_data(global_token_counter, token_ledger, num_heads) -> dict:
    """Fast-attention variant: ledger tracking is not available.

    Parameters mirror the cumulative variant's signature for interface parity,
    but are unused — this function always returns an empty dict.
    """
    warnings.warn(
        "get_ledger_data() is not supported in the fast-attention module. "
        "Use the cumulative module for research analysis.",
        stacklevel=2,
    )
    return {}
