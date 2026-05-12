"""
kv_cache/rerotation.py
----------------------
Utility functions for RoPE Key Rerotation.
"""

import torch
from .helpers import rotate_half

def unrotate_keys(keys, cos, sin):
    """Apply RoPE inverse: keys_unrotated = RoPE^-1(keys)"""
    return (keys * cos) + (rotate_half(keys) * (-sin))

def rerotate_keys(keys, cos_new, sin_new):
    """Apply fresh RoPE to unrotated keys."""
    return (keys * cos_new) + (rotate_half(keys) * sin_new)

def unrotate_keys_with_positions(key_states, rotary_emb, positions):
    """
    Un-rotate keys using their current RoPE positions.
    
    Parameters:
    - key_states: tensor ending in [..., head_dim]
    - rotary_emb: the RoPE module
    - positions: LongTensor of the same shape as key_states (excluding head_dim)
    """
    if positions.numel() == 0:
        return key_states
        
    max_pos = positions.max().item()
    cos, sin = rotary_emb(key_states, seq_len=max_pos + 1)
    
    cos_pos = cos[positions]
    sin_pos = sin[positions]
    
    return unrotate_keys(key_states, cos_pos, sin_pos)

def rerotate_cache_keys(key_states, rotary_emb, original_positions, new_seq_len):
    """
    Full pipeline: un-rotate keys from original positions, 
    re-rotate with contiguous [0..new_seq_len-1] positions.
    
    Parameters:
    - key_states: [1, num_heads, new_seq_len, head_dim]
    - rotary_emb: the Llama3RotaryEmbedding or HFRopeWrapper module
    - original_positions: [num_heads, new_seq_len] LongTensor of original position indices
    - new_seq_len: length of the new cache
    """
    if original_positions.numel() == 0:
        return key_states
        
    # 1. Un-rotate
    # original_positions is [num_heads, new_seq_len]. 
    # To broadcast with key_states [1, num_heads, new_seq_len, head_dim], 
    # we need positions to be [1, num_heads, new_seq_len]
    unrotated_keys = unrotate_keys_with_positions(
        key_states, 
        rotary_emb, 
        original_positions.unsqueeze(0)
    )
    
    # 2. Re-rotate with contiguous positions [0..new_seq_len-1]
    # We only need to compute this once
    max_pos = max(original_positions.max().item(), new_seq_len - 1)
    cos, sin = rotary_emb(key_states, seq_len=max_pos + 1)
    
    cos_new = cos[:new_seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, new_seq_len, head_dim]
    sin_new = sin[:new_seq_len].unsqueeze(0).unsqueeze(0)
    
    return rerotate_keys(unrotated_keys, cos_new, sin_new)
