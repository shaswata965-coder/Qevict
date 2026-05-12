"""
kv_cache/eviction.py
--------------------
Eviction and span-lookup helpers for the Sticky KV-cache.

Extracted verbatim from four methods of STICKYKVCache_LayerWise
(sticky_kv_logic_cummulative.py):

    _find_logical_window_span                        lines 270–287
    _gather_window_from_current_kv                   lines 289–297
    _evict_from_window_scores                        lines 1239–1274
    _create_mask_and_evict_from_kv_cache_prompt_stage lines 1276–1322

Narrow API: each function receives only the specific tensors / scalars it
actually reads or mutates, rather than the full cache object.
No logic is added, removed, or reordered.
"""

import torch


# ---------------------------------------------------------------------------
# 1.  Logical-window span lookup
# ---------------------------------------------------------------------------

def find_logical_window_span(
    logical_id_map: torch.Tensor,
    omega: int,
    h: int,
    wid_val,
    seq_len: int,
):
    """Return the (start, end) physical slice for logical window *wid_val* on
    head *h*, or ``None`` if the window is absent / partial.

    Parameters
    ----------
    logical_id_map : [num_heads, compressed_len] LongTensor
    omega          : window size (tokens per window)
    h              : head index
    wid_val        : logical window ID (int or scalar)
    seq_len        : current physical KV sequence length
    """
    positions = (logical_id_map[h] == int(wid_val)).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return None

    start = int(positions.min().item())
    end = int(positions.max().item()) + 1

    expected = torch.arange(start, end, device=positions.device)
    if positions.numel() != (end - start) or not torch.equal(positions, expected):
        raise RuntimeError(f"Logical window {wid_val} has non-contiguous physical positions")

    if end - start != omega:
        return None  # defer partial windows; do not promote them into sticky/q-cache

    if end > seq_len:
        return None
    return start, end


# ---------------------------------------------------------------------------
# 2.  Gather a single window's KV data from the current cache
# ---------------------------------------------------------------------------

def gather_window_from_current_kv(
    logical_id_map: torch.Tensor,
    omega: int,
    past_key_values,
    h: int,
    wid_val,
    *,
    seq_len: int,
):
    """Return ``(k_slice, v_slice)`` for logical window *wid_val* on head *h*,
    or ``None`` if the window cannot be located.

    Parameters
    ----------
    logical_id_map  : [num_heads, compressed_len] LongTensor
    omega           : window size (tokens per window)
    past_key_values : tuple of (K, V) tensors, shape [1, H, S, D]
    h               : head index
    wid_val         : logical window ID
    seq_len         : current physical KV sequence length (keyword-only)
    """
    span = find_logical_window_span(logical_id_map, omega, h, wid_val, seq_len)
    if span is None:
        return None
    start, end = span
    return (
        past_key_values[0][0, h, start:end],
        past_key_values[1][0, h, start:end],
    )


# ---------------------------------------------------------------------------
# 3.  Prefill eviction — window score selection
# ---------------------------------------------------------------------------

def evict_from_window_scores(
    window_scores: torch.Tensor,
    k_windows: int,
    q_windows_count: int,
):
    """Select top-k sticky windows from *window_scores*, mutate it in-place,
    and optionally return the top-q loser windows for the quantised side-cache.

    Parameters
    ----------
    window_scores   : [num_heads, max_windows, 3] float32 — mutated in-place.
                      Dim 2: [Cumulative Score, Current Window ID, Original Window ID]
    k_windows       : number of BF16 sticky windows to keep
    q_windows_count : number of q-cache (quantised) windows to capture

    Returns
    -------
    q_loser_ids    : [num_heads, q_count] float32, or None
    q_loser_scores : [num_heads, q_count] float32, or None
    """
    valid_mask = ~torch.isnan(window_scores[:, :, 1])
    scores = torch.where(
        valid_mask,
        window_scores[:, :, 0],
        torch.tensor(float("-inf"), device=window_scores.device),
    )
    ids, orig_ids = window_scores[:, :, 1], window_scores[:, :, 2]
    
    curr_k = min(k_windows, int(valid_mask.sum(dim=1).min().item()))
    
    top_v, top_i = torch.topk(scores, curr_k, dim=1, largest=True)
    kept_ids, kept_orig = torch.gather(ids, 1, top_i), torch.gather(
        orig_ids, 1, top_i
    )
    
    # Capture top-q losers BEFORE overwriting window_scores
    q_loser_ids = None
    q_loser_scores = None
    if q_windows_count > 0:
        total_valid = int(valid_mask.sum(dim=1).min().item())
        num_losers = total_valid - curr_k
        if num_losers > 0:
            loser_scores = scores.clone()
            loser_scores.scatter_(1, top_i, float("-inf"))
            q_count = min(q_windows_count, num_losers)
            q_top_v, q_top_i = torch.topk(loser_scores, q_count, dim=1, largest=True)
            q_loser_ids = torch.gather(ids, 1, q_top_i)
            q_loser_scores = q_top_v
    
    sort_idx = torch.argsort(kept_ids, dim=1)
    window_scores.fill_(float("nan"))
    window_scores[:, :curr_k, 0] = torch.gather(top_v, 1, sort_idx)
    window_scores[:, :curr_k, 1] = torch.gather(kept_ids, 1, sort_idx)
    window_scores[:, :curr_k, 2] = torch.gather(kept_orig, 1, sort_idx)
    return q_loser_ids, q_loser_scores


# ---------------------------------------------------------------------------
# 4.  Prompt-stage physical KV eviction
# ---------------------------------------------------------------------------

def create_mask_and_evict_from_kv_cache_prompt_stage(
    past_key_values,
    attn_scores: torch.Tensor,
    local_start_idx: int,
    *,
    k_seq_dim: int,
    window_scores: torch.Tensor,
    k_windows: int,
    sink_indices: torch.Tensor,
    window_to_token_map: torch.Tensor,
    num_heads: int,
):
    """Build a dense survivor KV cache after prompt-stage eviction.

    Parameters (narrow tensors only)
    ----------------------------------
    past_key_values   : (K, V) tuple, shapes [1, H, S, D]
    attn_scores       : attention weight tensor (unused beyond shape; kept for
                        API parity with the original signature)
    local_start_idx   : physical index where the local zone begins
    k_seq_dim         : sequence dimension of the KV tensors (always 2)
    window_scores     : [H, max_windows, 3] — already updated by evict_from_window_scores
    k_windows         : number of sticky windows kept
    sink_indices      : [sink_tokens] LongTensor of protected sink positions
    window_to_token_map : [max_windows, omega] LongTensor — logical→physical map
    num_heads         : number of KV heads

    Returns
    -------
    updated_kv   : (new_K, new_V) tuple with evicted tokens removed
    final_indices: [num_heads, N_survivors] LongTensor of retained physical indices
    """
    seq_len, head_dim = (
        past_key_values[0].size(k_seq_dim),
        past_key_values[0].shape[-1],
    )
    
    device = window_scores.device
    
    sinks = sink_indices.unsqueeze(0).expand(num_heads, -1)
    
    # FIX (C3): Only include genuinely valid (non-NaN) window score entries.
    # Previously, invalid NaN slots were zeroed to token index 0 via multiply-by-0,
    # inserting a phantom token-0 into the survivor set when sink_tokens == 0.
    raw_w = window_scores[:, :k_windows, 1]
    valid_w_mask = ~torch.isnan(raw_w)
    valid_k = int(valid_w_mask.all(dim=0).sum().item())
    if valid_k > 0:
        sticky_w = window_scores[:, :valid_k, 1].long()
        all_window_tokens = window_to_token_map[sticky_w]
        window_tokens = all_window_tokens.view(num_heads, -1)
    else:
        window_tokens = torch.zeros(num_heads, 0, device=device, dtype=torch.long)
    
    local_start = local_start_idx
    if local_start < seq_len:
        local_zone = torch.arange(local_start, seq_len, device=device).unsqueeze(0).expand(num_heads, -1)
        all_indices = torch.cat([sinks, window_tokens, local_zone], dim=1)
    else:
        all_indices = torch.cat([sinks, window_tokens], dim=1)
        
    all_indices_clamped = torch.clamp(all_indices, 0, seq_len - 1)
    
    # FIX (Issue 2): Remove safe_len min-truncation and diff deduplication.
    # Since sinks, window_tokens, and local_zone are mutually exclusive logically,
    # there are no duplicates. Sorting provides the exact dense timeline.
    sorted_all, _ = torch.sort(all_indices_clamped, dim=1)   # [H, N]
    final_indices = sorted_all
    
    gather_idx = (
        final_indices
        .unsqueeze(-1)
        .expand(-1, -1, head_dim)
    )
    return (
        torch.gather(past_key_values[0][0], 1, gather_idx).unsqueeze(0),
        torch.gather(past_key_values[1][0], 1, gather_idx).unsqueeze(0),
    ), final_indices
