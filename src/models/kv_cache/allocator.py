"""
kv_cache/allocator.py
---------------------
CFO (Cache-First Order) sequential budget allocator for the Sticky KV-cache.

REFACTORED: The original `update_k_win_and_local_num(cache, ...)` mutated the
full cache object directly, creating tight coupling.  It is replaced by a pure
function `compute_budget(...)` that receives only the required scalars and
returns a BudgetResult namedtuple.  The coordinator calls this and distributes
the fields to its managers.

The old function is kept as a deprecated shim for any code that still calls it
directly, but will be removed in a future cleanup.
"""

from collections import namedtuple

BudgetResult = namedtuple(
    "BudgetResult",
    ["local_num", "k_windows", "q_windows_count", "q_num", "quant_bytes_len"],
)


def compute_budget(
    total_cache_ratio: int,
    sink_tokens: int,
    use_fixed_local_tokens: bool,
    local_num_tokens: int,
    local_cache_ratio: int,
    q_ratio: int,
    quant_bit_width: int,
    head_dim: int,
    omega: int,
    new_tokens: int,
    max_tokens: int,
    layer_idx: int,
) -> BudgetResult:
    """Compute the CFO token-budget allocation.

    Parameters
    ----------
    total_cache_ratio   : % of (new_tokens + max_tokens) kept in cache.
    sink_tokens         : number of always-protected sink tokens.
    use_fixed_local_tokens : if True, use local_num_tokens directly.
    local_num_tokens    : fixed local-zone size (used when flag is True).
    local_cache_ratio   : % of remaining budget for local zone (when not fixed).
    q_ratio             : % of sticky budget reserved for quantised side-cache.
    quant_bit_width     : 8 (INT8) or 4 (packed INT4).
    head_dim            : attention head dimension (for compression ratio).
    omega               : window size in tokens.
    new_tokens          : number of tokens in the prompt.
    max_tokens          : max generation tokens (from GENERATION_CONFIG).
    layer_idx           : layer index (for warning messages only).

    Returns
    -------
    BudgetResult namedtuple with fields:
        local_num, k_windows, q_windows_count, q_num, quant_bytes_len
    """
    total_token_budget = (new_tokens + max_tokens) * total_cache_ratio // 100

    # --- SEQUENTIAL CARVING ALLOCATOR ---
    # Priority 1: Sinks (always kept)
    remaining = max(0, total_token_budget - sink_tokens)

    # Priority 2: Local zone
    if use_fixed_local_tokens:
        target_local_tokens = local_num_tokens
    else:
        target_local_tokens = (total_token_budget * local_cache_ratio) // 100
    local_num = min(target_local_tokens, remaining)
    remaining = max(0, remaining - local_num)

    # Priority 3+4: Split remaining between BF16 sticky and INT4/8 q-cache.
    bf16_bytes = 2 * head_dim
    quant_bytes = head_dim if quant_bit_width == 8 else (head_dim / 2.0)
    compression_ratio = bf16_bytes / quant_bytes

    bf16_target = (remaining * (100 - q_ratio)) // 100
    q_mem_target = remaining - bf16_target          # complement avoids double rounding
    q_int4_target = round(q_mem_target * compression_ratio)  # round() avoids int() floor on e.g. 507.999

    # Ceiling division: absorbs partial window rather than recycling to local.
    k_windows = -(-bf16_target // omega)
    q_windows_count = -(-q_int4_target // omega)

    # BF16-equivalent memory consumed by q-cache
    q_mem_used = int((q_windows_count * omega) / compression_ratio)
    q_num = q_mem_used

    if k_windows == 0:
        print(
            f"WARNING [Layer {layer_idx}]: k_windows=0 — insufficient budget for "
            f"sticky windows (budget={total_token_budget}, local={local_num}, "
            f"sink={sink_tokens}, q_windows={q_windows_count}). "
            f"Eviction is effectively disabled."
        )

    quant_bytes_len = head_dim if quant_bit_width == 8 else (head_dim // 2)

    return BudgetResult(
        local_num=local_num,
        k_windows=k_windows,
        q_windows_count=q_windows_count,
        q_num=q_num,
        quant_bytes_len=quant_bytes_len,
    )


# ---------------------------------------------------------------------------
# Deprecated shim — kept for backward compatibility
# ---------------------------------------------------------------------------

def update_k_win_and_local_num(cache, new_tokens, max_tokens):
    """DEPRECATED: Use compute_budget() instead.

    Reads config from *cache* and writes results back in-place.
    Will be removed in a future cleanup pass.
    """
    result = compute_budget(
        total_cache_ratio=cache.total_cache_ratio,
        sink_tokens=cache.sink_tokens,
        use_fixed_local_tokens=cache.use_fixed_local_tokens,
        local_num_tokens=cache.local_num_tokens,
        local_cache_ratio=cache.local_cache_ratio,
        q_ratio=cache.q_ratio,
        quant_bit_width=cache.quant_bit_width,
        head_dim=cache.head_dim,
        omega=cache.omega,
        new_tokens=new_tokens,
        max_tokens=max_tokens,
        layer_idx=cache.layer_idx,
    )
    cache.local_num = result.local_num
    cache.k_windows = result.k_windows
    cache.q_windows_count = result.q_windows_count
    cache.q_num = result.q_num
    cache._quant_bytes_len = result.quant_bytes_len
