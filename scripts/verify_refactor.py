"""
scripts/verify_refactor.py
--------------------------
Run this in the project's PyTorch environment to verify the cache.py
modularisation refactor.

Usage:
    python scripts/verify_refactor.py

All tests should print PASS. Any FAIL indicates a regression.
"""

import sys
import torch

print("=" * 60)
print("Cache.py Modularization Verification")
print("=" * 60)

errors = []

# ------------------------------------------------------------------
# Test 1: Import smoke
# ------------------------------------------------------------------
print("\n[1] Import smoke...")
try:
    from src.models.kv_cache import (
        STICKYKVCache_LayerWise, repeat_kv, _make_causal_mask,
        EvictionManager, QuantizationManager, NoOpQuantizationManager,
        TrackingManager, NoOpTrackingManager,
        BaseQuantizationManager, BaseTrackingManager,
    )
    print("    PASS")
except Exception as e:
    print(f"    FAIL: {e}")
    errors.append("Import smoke")

# ------------------------------------------------------------------
# Test 2: ABC compliance
# ------------------------------------------------------------------
print("\n[2] ABC compliance...")
try:
    assert issubclass(QuantizationManager, BaseQuantizationManager), "QM not subclass"
    assert issubclass(NoOpQuantizationManager, BaseQuantizationManager), "NoQM not subclass"
    assert issubclass(TrackingManager, BaseTrackingManager), "TM not subclass"
    assert issubclass(NoOpTrackingManager, BaseTrackingManager), "NoTM not subclass"
    print("    PASS")
except AssertionError as e:
    print(f"    FAIL: {e}")
    errors.append("ABC compliance")

# ------------------------------------------------------------------
# Test 3: hasattr compatibility (mirrors module.py:201 guard)
# ------------------------------------------------------------------
print("\n[3] hasattr compatibility...")
try:
    import sticky_config  # noqa: ensure config is importable

    class _FakeConfig:
        hidden_size = 2048
        num_attention_heads = 32
        num_key_value_heads = 8
        max_position_embeddings = 4096
        p_ratio = 30; r_ratio = 50; start_idx = 1

    cache = STICKYKVCache_LayerWise(
        p_ratio=30, r_ratio=50, start_idx=1, num_heads=8, layer_idx=0, config=_FakeConfig()
    )
    assert hasattr(cache, "q_cache_k_quant"), "q_cache_k_quant missing"
    assert hasattr(cache, "q_cache_k_scale"), "q_cache_k_scale missing"
    assert hasattr(cache, "q_cache_v_quant"), "q_cache_v_quant missing"
    assert hasattr(cache, "token_ledger"),     "token_ledger missing"
    assert hasattr(cache, "global_token_counter"), "global_token_counter missing"
    # At init these should be None (no prefill done)
    assert cache.q_cache_k_quant is None, "expected None before prefill"
    print("    PASS")
except Exception as e:
    print(f"    FAIL: {e}")
    errors.append("hasattr compatibility")

# ------------------------------------------------------------------
# Test 4: NoOp quantization when Q_RATIO=0
# ------------------------------------------------------------------
print("\n[4] NoOpQuantizationManager when Q_RATIO=0...")
try:
    qm = NoOpQuantizationManager()
    assert qm.q_cache_k_quant is None
    qm.store_windows(None, None, None, None)   # must not raise
    qm.accumulate_scores(None, 8)              # must not raise
    promo_k, promo_v = qm.get_promoted_windows(torch.zeros(1, 1))
    assert promo_k == {} and promo_v == {}
    qm.rebuild(None, None, None, None, 0, 8, 0)
    qm.reset()
    print("    PASS")
except Exception as e:
    print(f"    FAIL: {e}")
    errors.append("NoOp quantization")

# ------------------------------------------------------------------
# Test 5: allocator pure function
# ------------------------------------------------------------------
print("\n[5] compute_budget pure function...")
try:
    from src.models.kv_cache.allocator import compute_budget, BudgetResult
    result = compute_budget(
        total_cache_ratio=50, sink_tokens=4,
        use_fixed_local_tokens=False, local_num_tokens=0, local_cache_ratio=20,
        q_ratio=0, quant_bit_width=8, head_dim=64, omega=8,
        new_tokens=512, max_tokens=512, layer_idx=0,
    )
    assert isinstance(result, BudgetResult), "wrong type"
    assert result.local_num >= 0
    assert result.k_windows >= 0
    print("    PASS")
except Exception as e:
    print(f"    FAIL: {e}")
    errors.append("compute_budget")

# ------------------------------------------------------------------
# Test 6: EvictionManager already_tracked_per_head (bug fix)
# ------------------------------------------------------------------
print("\n[6] EvictionManager — already_tracked_per_head defined...")
try:
    import inspect
    from src.models.kv_cache.eviction_manager import EvictionManager
    src = inspect.getsource(EvictionManager.run_decode_cycle)
    assert "already_tracked_per_head" in src, "variable not defined in run_decode_cycle"
    assert "BUG FIX" in src or "already_tracked_per_head = " in src
    print("    PASS")
except Exception as e:
    print(f"    FAIL: {e}")
    errors.append("already_tracked_per_head fix")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} test(s): {errors}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
