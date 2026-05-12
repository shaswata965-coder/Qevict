"""
attention/module.py
-------------------
STICKYLlamaAttention — standard PyTorch SDPA backend.

The class structure and forward() logic are identical to the original
sticky_llama_attention.py.  This module simply delegates to the shared
sub-modules so each concern lives in exactly one place:

  rope.py      →  Llama3RotaryEmbedding, init_rope()
  ops.py       →  compute_main_logits, apply_prefill_causal_mask,
                  compute_qcache_joint_softmax, compute_standard_softmax
  module.py    →  STICKYLlamaAttention (this file)
"""

import torch
from torch import nn
from typing import Optional, Tuple

# DynamicCache compatibility (transformers >= 4.43)
try:
    from transformers.cache_utils import DynamicLayer as _DynamicLayer
except ImportError:
    _DynamicLayer = None  # older transformers — legacy tuple path only

from src.models.sticky_kv_logic_cummulative import (
    _make_causal_mask,
    apply_rotary_pos_emb_single,
    STICKYKVCache_LayerWise,
)
from .rope import Llama3RotaryEmbedding, init_rope  # noqa: F401 (re-exported for convenience)
from .ops import (
    compute_main_logits,
    apply_prefill_causal_mask,
    compute_qcache_joint_softmax,
    compute_standard_softmax,
)


class STICKYLlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) not divisible by num_heads ({self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope()

        self.kv_cache = STICKYKVCache_LayerWise(
            p_ratio=config.p_ratio,
            r_ratio=config.r_ratio,
            start_idx=config.start_idx,
            num_heads=config.num_key_value_heads,
            layer_idx=layer_idx,
            config=config,
        )
        
        self.kv_cache.set_rotary_emb(self.rotary_emb)

    # ------------------------------------------------------------------
    # RoPE initialisation — delegates to the shared factory in rope.py
    # ------------------------------------------------------------------

    def _init_rope(self):
        self.rotary_emb = init_rope(
            self.config,
            self.head_dim,
            self.max_position_embeddings,
            self.rope_theta,
        )

    # ------------------------------------------------------------------
    # Cache reset
    # ------------------------------------------------------------------

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):

        bsz, q_len, _ = hidden_states.size()

        # --- HF DynamicCache Compatibility ---
        if past_key_value is None and "past_key_values" in kwargs:
            past_key_value = kwargs["past_key_values"]

        is_hf_cache = False
        hf_cache_obj = None
        if past_key_value is not None and not isinstance(past_key_value, tuple):
            is_hf_cache = True
            hf_cache_obj = past_key_value
            # New API (transformers >= 4.43): cache.layers[layer_idx].keys / .values
            if hasattr(hf_cache_obj, "layers") and len(hf_cache_obj.layers) > self.layer_idx:
                layer_cache = hf_cache_obj.layers[self.layer_idx]
                if getattr(layer_cache, "is_initialized", False):
                    past_key_value = (layer_cache.keys, layer_cache.values)
                else:
                    past_key_value = None
            else:
                past_key_value = None

        # 1. Update position_ids
        phys_past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                phys_past_len, phys_past_len + q_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)

        # 2. Project Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. Causal Masking
        phys_past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        attention_mask = None
        if q_len == 1:
            attention_mask = _make_causal_mask(
                bsz, q_len, phys_past_len, query_states.dtype, query_states.device
            )

        # 4. Rotary Positional Embeddings
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            kv_seq_len = key_states.shape[-2] + phys_past_len
            cos, sin = self.rotary_emb(
                value_states, seq_len=max(kv_seq_len, position_ids.max().item() + 1)
            )
        
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        # 5. KV Cache Concatenation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 6. Attention logits (GQA or MHA)
        main_logits = compute_main_logits(
            query_states, key_states,
            bsz, self.num_heads, self.num_key_value_heads,
            self.num_key_value_groups, q_len, self.head_dim,
        )

        # 7. Apply mask
        if attention_mask is not None:
            main_logits = main_logits + attention_mask
        elif q_len > 1:
            # Apply causal mask in-place to avoid materializing O(N^2) mask tensor
            apply_prefill_causal_mask(main_logits, q_len, phys_past_len)

        # 8. Softmax + output  (q-cache joint-softmax or standard path)
        # --- QUANT Q-CACHE: Joint softmax with dequantized quantized cache ---
        q_scores_for_cache = None
        if (q_len == 1
                and hasattr(self.kv_cache, "q_cache_k_quant")
                and self.kv_cache.q_cache_k_quant is not None):
            attn_output, scores_for_cache, q_scores_for_cache, attn_weights_for_output = \
                compute_qcache_joint_softmax(
                    query_states, main_logits, value_states, self.kv_cache,
                    bsz, self.num_heads, self.num_key_value_heads,
                    self.num_key_value_groups, q_len, self.head_dim,
                )
        else:
            # Standard path (no q-cache or prefill)
            attn_output, scores_for_cache, attn_weights_for_output = \
                compute_standard_softmax(
                    query_states, main_logits, value_states,
                    bsz, self.num_heads, self.num_key_value_heads,
                    self.num_key_value_groups, q_len, self.head_dim,
                )

        # 9. Custom Sticky KV Cache Eviction
        # PASS FULL ATTENTION SCORES for Pre-fill Research Analysis
        past_key_value = self.kv_cache(
            past_key_value,
            scores_for_cache.detach(),
            full_attn_scores=attn_weights_for_output.detach(),
            q_len=q_len,
            q_attn_scores=q_scores_for_cache.detach() if q_scores_for_cache is not None else None,
        )

        # 10. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if use_cache and is_hf_cache and past_key_value is not None:
            # New API (transformers >= 4.43): write back to cache.layers[idx].keys/.values
            if hasattr(hf_cache_obj, "layers") and _DynamicLayer is not None:
                while len(hf_cache_obj.layers) <= self.layer_idx:
                    hf_cache_obj.layers.append(_DynamicLayer())
                layer_cache = hf_cache_obj.layers[self.layer_idx]
                layer_cache.keys = past_key_value[0]
                layer_cache.values = past_key_value[1]
                layer_cache.is_initialized = True
            past_key_value = hf_cache_obj

        # Always return exactly 2 elements, as expected by the latest transformers API
        return attn_output, (attn_weights_for_output if output_attentions else None)
