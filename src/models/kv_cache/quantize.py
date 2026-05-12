"""
kv_cache/quantize.py
--------------------
INT8 / packed-INT4 quantisation and dequantisation for the Sticky KV-cache.

Extracted verbatim from the three @staticmethod blocks in
STICKYKVCache_LayerWise (sticky_kv_logic_cummulative.py, lines 1184–1237).
No logic is added, removed, or reordered.

The leading underscore has been dropped from the names to reflect that these
are now module-level functions rather than private class members.  The class
re-exports them under the original underscore names via thin @staticmethod
wrappers so all existing call sites (including ops.py) continue to work.
"""

import torch


def quantize_k_per_window(tensor, bit_width=8):
    """Quantize K cache: per-channel per-window, with RoPE-paired dimension tying.
    Args:
        tensor: [H, W, omega, D] fp16 — already has RoPE applied
    Returns: (uint8_tensor[H,W,omega,D], scale[H,W,1,D], zp[H,W,1,D])
    """
    # Per-channel: reduce across token dim (omega) → [H, W, 1, D]
    t_min = tensor.amin(dim=2, keepdim=True)
    t_max = tensor.amax(dim=2, keepdim=True)
    half_d = tensor.shape[-1] // 2
    t_min_h1, t_min_h2 = t_min[..., :half_d], t_min[..., half_d:]
    t_max_h1, t_max_h2 = t_max[..., :half_d], t_max[..., half_d:]
    t_min_tied = torch.min(t_min_h1, t_min_h2)
    t_max_tied = torch.max(t_max_h1, t_max_h2)
    t_min = torch.cat([t_min_tied, t_min_tied], dim=-1)
    t_max = torch.cat([t_max_tied, t_max_tied], dim=-1)
    
    if bit_width == 4:
        scale = torch.clamp((t_max - t_min) / 15.0, min=1e-8)
        quantized = torch.round((tensor - t_min) / scale).clamp(0, 15).to(torch.uint8)
        packed = (quantized[..., 0::2] << 4) | quantized[..., 1::2]
        return packed, scale.to(tensor.dtype), t_min.to(tensor.dtype)
    else:
        scale = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
        quantized = torch.round((tensor - t_min) / scale).clamp(0, 255).to(torch.uint8)
        return quantized, scale.to(tensor.dtype), t_min.to(tensor.dtype)


def quantize_v_per_window(tensor, bit_width=8):
    """Quantize V cache: per-token per-window."""
    t_min = tensor.amin(dim=3, keepdim=True)
    t_max = tensor.amax(dim=3, keepdim=True)
    if bit_width == 4:
        scale = torch.clamp((t_max - t_min) / 15.0, min=1e-8)
        quantized = torch.round((tensor - t_min) / scale).clamp(0, 15).to(torch.uint8)
        packed = (quantized[..., 0::2] << 4) | quantized[..., 1::2]
        return packed, scale.to(tensor.dtype), t_min.to(tensor.dtype)
    else:
        scale = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
        quantized = torch.round((tensor - t_min) / scale).clamp(0, 255).to(torch.uint8)
        return quantized, scale.to(tensor.dtype), t_min.to(tensor.dtype)


def dequantize_from_quant(quant_tensor, scale, zero_point, bit_width=8):
    """Dequantize int8 or packed int4 tensor back to fp16."""
    if bit_width == 4:
        q_even = (quant_tensor >> 4) & 0x0F
        q_odd = quant_tensor & 0x0F
        unpacked = torch.stack((q_even, q_odd), dim=-1)
        unpacked = unpacked.view(*quant_tensor.shape[:-1], -1)
        return unpacked.to(scale.dtype) * scale + zero_point
    else:
        return quant_tensor.to(scale.dtype) * scale + zero_point
