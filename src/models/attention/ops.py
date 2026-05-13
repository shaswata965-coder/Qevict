"""
attention/ops.py
----------------
Pure, stateless tensor operations for the SDPA (standard) attention backend.

These functions are extracted verbatim from STICKYLlamaAttention.forward()
(sticky_llama_attention.py) — no logic is added, removed, or reordered.
They are also reused by the Flash-Attention backend's decoding step
(module_flash.py), since the decoding path is identical in both backends.
"""

import math
from typing import Tuple

import torch



# ---------------------------------------------------------------------------
# 1. Attention logit computation  (GQA / MHA unified)
# ---------------------------------------------------------------------------

def compute_main_logits(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    bsz: int,
    num_heads: int,
    num_kv_heads: int,
    num_kv_groups: int,
    q_len: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute raw (pre-softmax) attention logits, supporting both MHA and GQA."""
    if num_heads != num_kv_heads:
        q_grouped = query_states.reshape(
            bsz, num_kv_heads, num_kv_groups, q_len, head_dim
        )
        main_logits = torch.matmul(
            q_grouped, key_states.transpose(2, 3).unsqueeze(2)
        ) / math.sqrt(head_dim)
        main_logits = main_logits.reshape(bsz, num_heads, q_len, -1)
    else:
        main_logits = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
    return main_logits


# ---------------------------------------------------------------------------
# 2. In-place prefill causal mask  (avoids O(N²) tensor allocation)
# ---------------------------------------------------------------------------

def apply_prefill_causal_mask(
    main_logits: torch.Tensor,
    q_len: int,
    phys_past_len: int,
) -> torch.Tensor:
    """Apply causal mask in-place during prefill.

    Avoids materialising a full O(N²) mask tensor by using index broadcasting.
    Modifies ``main_logits`` in-place and returns it for convenience.
    """
    mask_val = torch.finfo(main_logits.dtype).min
    i_idx = torch.arange(q_len, device=main_logits.device).unsqueeze(1)
    j_idx = torch.arange(q_len + phys_past_len, device=main_logits.device).unsqueeze(0)
    main_logits.masked_fill_(i_idx + phys_past_len < j_idx, mask_val)
    return main_logits


# ---------------------------------------------------------------------------
# 3. Q-cache joint-softmax path  (decoding only)
# ---------------------------------------------------------------------------

def compute_qcache_joint_softmax(
    query_states: torch.Tensor,
    main_logits: torch.Tensor,
    value_states: torch.Tensor,
    kv_cache,
    bsz: int,
    num_heads: int,
    num_kv_heads: int,
    num_kv_groups: int,
    q_len: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joint softmax over the main BF16 KV cache and the dequantized q-cache.

    Used only during the decoding step (q_len == 1) when a quantized side-cache
    is active.

    Returns
    -------
    attn_output          : Tensor (bsz, num_heads, q_len, head_dim)
    scores_for_cache     : Main-cache attention weights, kv-head averaged
    q_scores_for_cache   : Q-cache attention weights, kv-head averaged
    attn_weights_for_output : Main-only weights (no q-cache dims) for output_attentions
    """
    # Dequantize on the fly
    q_k = kv_cache._dequantize_from_quant(
        kv_cache.q_cache_k_quant,
        kv_cache.q_cache_k_scale,
        kv_cache.q_cache_k_zp,
        kv_cache.quant_bit_width,
    )
    q_v = kv_cache._dequantize_from_quant(
        kv_cache.q_cache_v_quant,
        kv_cache.q_cache_v_scale,
        kv_cache.q_cache_v_zp,
        kv_cache.quant_bit_width,
    )

    # Flatten W*omega → total_tokens: [H, W*omega, D]
    H, W, omega, D = q_k.shape
    q_k = q_k.reshape(H, W * omega, D).unsqueeze(0)
    q_v = q_v.reshape(H, W * omega, D).unsqueeze(0)

    if num_heads != num_kv_heads:
        q_grouped = query_states.reshape(
            bsz, num_kv_heads, num_kv_groups, q_len, head_dim
        )
        q_logits_grouped = torch.matmul(
            q_grouped, q_k.transpose(2, 3).unsqueeze(2)
        ) / math.sqrt(head_dim)
        q_logits = q_logits_grouped.reshape(bsz, num_heads, q_len, -1)
    else:
        q_logits = torch.matmul(query_states, q_k.transpose(2, 3)) / math.sqrt(head_dim)

    # Joint softmax over [main | q-cache]
    all_logits = torch.cat([main_logits, q_logits], dim=-1)
    attn_weights = all_logits.to(torch.float32)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(query_states.dtype)

    main_len = main_logits.size(-1)
    attn_weights_main = attn_weights[..., :main_len]
    attn_weights_q = attn_weights[..., main_len:]

    if num_heads != num_kv_heads:
        attn_main_grouped = attn_weights_main.reshape(
            bsz, num_kv_heads, num_kv_groups, q_len, -1
        )
        value_main_grouped = value_states.unsqueeze(2)
        out_main = torch.matmul(attn_main_grouped, value_main_grouped)

        attn_q_grouped = attn_weights_q.reshape(
            bsz, num_kv_heads, num_kv_groups, q_len, -1
        )
        out_q = torch.matmul(attn_q_grouped, q_v.unsqueeze(2))

        attn_output = (out_main + out_q).reshape(bsz, num_heads, q_len, head_dim)
        scores_for_cache = attn_weights_main.reshape(
            bsz, num_kv_heads, num_kv_groups, q_len, -1
        ).mean(dim=2)
        q_scores_for_cache = attn_weights_q.reshape(
            bsz, num_kv_heads, num_kv_groups, q_len, -1
        ).mean(dim=2)
    else:
        attn_output = (
            torch.matmul(attn_weights_main, value_states)
            + torch.matmul(attn_weights_q, q_v)
        )
        scores_for_cache = attn_weights_main
        q_scores_for_cache = attn_weights_q

    # FIX (C4): Track main-only attention weights for output_attentions.
    # The joint softmax attn_weights includes q-cache dimensions that
    # downstream consumers don't expect.
    attn_weights_for_output = attn_weights_main
    return attn_output, scores_for_cache, q_scores_for_cache, attn_weights_for_output


# ---------------------------------------------------------------------------
# 4. Standard softmax path  (prefill or decoding without q-cache)
# ---------------------------------------------------------------------------

def compute_standard_softmax(
    query_states: torch.Tensor,
    main_logits: torch.Tensor,
    value_states: torch.Tensor,
    bsz: int,
    num_heads: int,
    num_kv_heads: int,
    num_kv_groups: int,
    q_len: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard softmax path — used during prefill and decoding when q-cache is inactive.

    Returns
    -------
    attn_output      : Tensor (bsz, num_heads, q_len, head_dim)
    scores_for_cache : Attention weights averaged to kv-head granularity
    attn_weights     : Full attention weights (for output_attentions)
    """
    # Standard path (no q-cache or prefill)
    attn_weights = main_logits.to(torch.float32)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(query_states.dtype)

    if num_heads != num_kv_heads:
        attn_grouped = attn_weights.reshape(
            bsz, num_kv_heads, num_kv_groups, q_len, -1
        )
        attn_output = torch.matmul(
            attn_grouped, value_states.unsqueeze(2)
        ).reshape(bsz, num_heads, q_len, head_dim)
        scores_for_cache = attn_grouped.mean(dim=2)
    else:
        attn_output = torch.matmul(attn_weights, value_states)
        scores_for_cache = attn_weights

    return attn_output, scores_for_cache, attn_weights
