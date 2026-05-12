"""
attention/ops_flash.py
----------------------
Flash-Attention v2 specific ops for the prefill (prompt) stage.

These are extracted verbatim from STICKYLlamaAttention.forward()
in sticky_llama_attention_fast_attention.py — no logic changes.
The decoding path in that backend reuses the shared ops.py functions.
"""

import math
from typing import Tuple

import torch
from flash_attn import flash_attn_func


# ---------------------------------------------------------------------------
# 1. Flash Attention prefill output
# ---------------------------------------------------------------------------

def prefill_flash_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Run Flash-Attention v2 for the prefill (prompt) stage.

    Expects tensors in (batch_size, num_heads, seq_len, head_dim) layout.
    Returns attn_output in the same layout.

    Using original un-repeated k/v takes advantage of FA2's native GQA support.
    """
    # Reformat to (batch_size, seq_len, num_heads, head_dim) for flash_attn_func
    q_fa = query_states.transpose(1, 2)
    k_fa = key_states.transpose(1, 2)
    v_fa = value_states.transpose(1, 2)

    attn_output = flash_attn_func(
        q_fa,
        k_fa,
        v_fa,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )
    # Reformat back to (batch_size, num_heads, seq_len, head_dim)
    return attn_output.transpose(1, 2)


# ---------------------------------------------------------------------------
# 2. Chunked prefill tracking scores  (memory-efficient, avoids O(N²) alloc)
# ---------------------------------------------------------------------------

def compute_chunked_prefill_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    q_len: int,
    phys_past_len: int,
    num_kv_heads: int,
    num_heads: int,
    num_kv_groups: int,
    head_dim: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute chunked attention scores used for cache eviction during prefill.

    FIX A+E+F: Uses un-repeated key_states with grouped matmul to avoid
    materialising key_states_rep (~134MB/layer). Processes queries in chunks
    of ``chunk_size`` to stay within memory budget.

    Returns accumulated_scores of shape (bsz, num_kv_heads, 1, kv_seq_len)
    so the downstream cache logic receives a consistent 4-D tensor.
    """
    bsz = query_states.shape[0]
    kv_seq_len_full = key_states.shape[-2]

    # Accumulated scores: [bsz, num_kv_heads, kv_seq_len]
    accumulated_scores = torch.zeros(
        (bsz, num_kv_heads, kv_seq_len_full),
        device=query_states.device,
        dtype=torch.float32,
    )

    # Pre-generate key indices for dynamic mask chunking
    k_indices = torch.arange(kv_seq_len_full, device=query_states.device).unsqueeze(0)  # [1, kv_seq_len]

    # Pre-transpose keys once: [bsz, kv_heads, head_dim, kv_seq_len]
    key_states_t = key_states.transpose(-2, -1)

    for i in range(0, q_len, chunk_size):
        chunk_len = min(chunk_size, q_len - i)
        q_chunk = query_states[:, :, i:i + chunk_len, :]

        # Dynamically construct the causal mask block for this chunk (O(N) memory)
        q_indices = torch.arange(i, i + chunk_len, device=query_states.device).unsqueeze(1)  # [chunk_len, 1]
        # mask_chunk is causal: valid where key_idx <= query_idx + past_len
        mask_chunk = torch.full(
            (chunk_len, kv_seq_len_full),
            torch.finfo(query_states.dtype).min,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        mask_chunk.masked_fill_(k_indices <= (q_indices + phys_past_len), 0.0)

        # Expand mask to [1, 1, chunk_len, kv_seq_len]
        mask_chunk = mask_chunk.unsqueeze(0).unsqueeze(0)

        # FIX A+E+F: Grouped matmul with un-repeated keys.
        # Reshape queries [bsz, num_heads, chunk, D] → [bsz, kv_heads, groups, chunk, D]
        # Matmul broadcasts key_states_t over the group dim (no copy).
        if num_heads != num_kv_heads:
            q_grouped = q_chunk.reshape(
                bsz, num_kv_heads, num_kv_groups, chunk_len, head_dim
            )
            # [bsz, kv_heads, groups, chunk, D] x [bsz, kv_heads, 1, D, kv_len]
            # → [bsz, kv_heads, groups, chunk, kv_len]
            attn_chunk = torch.matmul(
                q_grouped, key_states_t.unsqueeze(2)
            ) / math.sqrt(head_dim)
            attn_chunk = attn_chunk + mask_chunk
            # Softmax per-head per-query position (dim=-1 = kv_seq_len)
            attn_chunk = torch.softmax(attn_chunk.to(torch.float32), dim=-1).to(query_states.dtype)
            # Average across query head groups → [bsz, kv_heads, chunk, kv_len]
            scores_for_cache_chunk = attn_chunk.mean(dim=2)
        else:
            attn_chunk = torch.matmul(q_chunk, key_states_t) / math.sqrt(head_dim)
            attn_chunk = attn_chunk + mask_chunk
            attn_chunk = torch.softmax(attn_chunk.to(torch.float32), dim=-1).to(query_states.dtype)
            scores_for_cache_chunk = attn_chunk

        # Sum over q_len dim → [bsz, kv_heads, kv_seq_len]
        accumulated_scores += scores_for_cache_chunk.sum(dim=2).to(torch.float32)
        del attn_chunk, scores_for_cache_chunk

    del key_states_t

    # Re-expand to [bsz, kv_heads, 1, kv_seq_len] so downstream cache logic
    # receives a consistent 4-D tensor (dim=2 acts as the accumulated q_len=1 metric)
    return accumulated_scores.unsqueeze(2)
