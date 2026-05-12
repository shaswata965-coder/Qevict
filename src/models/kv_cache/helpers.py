"""
kv_cache/helpers.py
-------------------
Pure, stateless tensor helpers shared across the Sticky KV-cache pipeline.

Extracted verbatim from sticky_kv_logic_cummulative.py (lines 1–54).
No logic is added, removed, or reordered.
"""

import math

import torch
from transformers.models.llama.modeling_llama import rotate_half


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Get dimensions from the hidden states (kv-cache)
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # If the number of query heads exactly matches the number of KV heads, no repetition needed
    if n_rep == 1:
        return hidden_states
    
    # Add a dimension for repeats and expand it across that dimension
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    
    # Flatten the repeated head dimension into the original head dimension
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(bsz, tgt_len, past_len, dtype, device):
    # Initialize a mask with the minimum possible float value (negative infinity equivalent)
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    
    # Create the condition for what to mask (causal, upper triangular masked)
    mask_cond = torch.arange(mask.size(-1), device=device)
    
    # Fill the lower triangular part with 0s (allowed attention part)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    
    # Convert mask to the expected data type
    mask = mask.to(dtype)
    
    # Connect past cache length to the causal mask by prefixing zeros
    if past_len > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_len, dtype=dtype, device=device), mask], dim=-1
        )
        
    # Expand to match batch size and number of heads
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_len)


def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim=1):
    # Select the rotary embeddings specifically for the position IDs of the given queries
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    # Apply standard RoPE (Rotary Position Embedding) formula
    return (q * cos) + (rotate_half(q) * sin)
