"""
kv_cache_fast_attention/allocator.py
--------------------------------------
Re-exports the shared CFO budget allocator from kv_cache.allocator.
"""

from src.models.kv_cache.allocator import compute_budget, update_k_win_and_local_num, BudgetResult  # noqa: F401

__all__ = ["compute_budget", "update_k_win_and_local_num", "BudgetResult"]
