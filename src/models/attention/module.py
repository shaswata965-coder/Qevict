"""
attention/module.py
-------------------
STICKYLlamaAttention — standard PyTorch SDPA backend.

Key design:
  - Works with ANY cache type HF passes: DynamicCache, StickyCache, or None.
  - Reads KV from cache.key_cache[layer_idx] / cache.value_cache[layer_idx].
  - After eviction, writes back directly into the same cache object.
  - Updates cache._seen_tokens at layer 0 so HF's generate loop computes
    correct cache_position and causal masks.
  - position_ids during decode are derived from global_token_counter (the
    TRUE sequence position), NOT from the compressed physical cache size.
"""

import torch
from torch import nn
from typing import Optional, Tuple

from src.models.sticky_kv_logic_cummulative import (
    _make_causal_mask,
    apply_rotary_pos_emb_single,
    STICKYKVCache_LayerWise,
)
from .rope import Llama3RotaryEmbedding, init_rope  # noqa: F401
from .ops import (
    compute_main_logits,
    apply_prefill_causal_mask,
    compute_qcache_joint_softmax,
    compute_standard_softmax,
)


def _read_kv_from_cache(cache_obj, layer_idx):
    """Extract (k, v) from any HF-compatible cache object.

    Returns (past_kv, cache_obj) where:
      past_kv   = (k_tensor, v_tensor) or None if empty/prefill
      cache_obj = the cache object to write back to (never None after first call)
    """
    if cache_obj is None:
        return None, None

    # Both DynamicCache and StickyCache have key_cache / value_cache lists
    if hasattr(cache_obj, "key_cache"):
        if len(cache_obj.key_cache) > layer_idx:
            k = cache_obj.key_cache[layer_idx]
            v = cache_obj.value_cache[layer_idx]
            if k.numel() > 0:
                return (k, v), cache_obj
        return None, cache_obj

    # Legacy tuple path (very old HF or direct testing)
    if isinstance(cache_obj, tuple) and len(cache_obj) >= 2:
        if isinstance(cache_obj[0], torch.Tensor):
            return cache_obj, None

    return None, cache_obj


def _write_kv_to_cache(cache_obj, layer_idx, evicted_kv, global_token_counter):
    """Write evicted KV back into the cache object in-place.

    Also updates _seen_tokens at layer 0 so HF computes correct positions.
    """
    if cache_obj is None or evicted_kv is None:
        return

    # Grow lists if needed
    while len(cache_obj.key_cache) <= layer_idx:
        cache_obj.key_cache.append(torch.empty(0))
        cache_obj.value_cache.append(torch.empty(0))

    cache_obj.key_cache[layer_idx] = evicted_kv[0]
    cache_obj.value_cache[layer_idx] = evicted_kv[1]

    # Update _seen_tokens on layer 0.  HF reads this to compute cache_position
    # and build causal masks.  It must reflect the TRUE total sequence length
    # (not the compressed physical cache size) so that:
    #   - prepare_inputs_for_generation slices input_ids correctly
    #   - _update_causal_mask builds a mask of the right shape
    if layer_idx == 0 and hasattr(cache_obj, "_seen_tokens"):
        cache_obj._seen_tokens = int(global_token_counter)


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

    def _init_rope(self):
        self.rotary_emb = init_rope(
            self.config, self.head_dim, self.max_position_embeddings, self.rope_theta,
        )

    def _clean_cache(self):
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Debug counter (layer 0 only)
        if self.layer_idx == 0 and not hasattr(self, '_dbg_count'):
            self._dbg_count = 0

        # ------------------------------------------------------------------
        # 1. Read KV from whatever cache type HF passed.
        #
        #   DynamicCache → read from key_cache[layer_idx]
        #   StickyCache  → read from key_cache[layer_idx] (same API)
        #   None         → prefill, no prior context
        # ------------------------------------------------------------------
        past_kv, cache_obj = _read_kv_from_cache(past_key_value, self.layer_idx)

        if self.layer_idx == 0 and self._dbg_count < 5:
            cache_type = type(past_key_value).__name__ if past_key_value is not None else 'None'
            cache_id = id(past_key_value) if past_key_value is not None else 0
            past_kv_shape = past_kv[0].shape if past_kv is not None else 'None'
            print(f"[DBG L0 step={self._dbg_count}] q_len={q_len} cache_type={cache_type} (id={cache_id}) past_kv_shape={past_kv_shape}", flush=True)

        # ------------------------------------------------------------------
        # 2. Position IDs — use TRUE global sequence position.
        #
        #    CRITICAL: the original sticky_llama_attention.py used
        #    self.kv_cache.global_token_counter, NOT the physical cache
        #    length.  Using phys_past_len causes wrong RoPE positions because
        #    the cache is compressed by eviction (phys < true sequence pos).
        # ------------------------------------------------------------------
        if past_kv is not None:
            global_past_len = int(self.kv_cache.global_token_counter.item())
            position_ids = torch.arange(
                global_past_len, global_past_len + q_len,
                dtype=torch.long, device=hidden_states.device,
            ).unsqueeze(0)
        elif position_ids is None:
            position_ids = torch.arange(
                0, q_len, dtype=torch.long, device=hidden_states.device,
            ).unsqueeze(0)

        # ------------------------------------------------------------------
        # 3. Project Q, K, V
        # ------------------------------------------------------------------
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ------------------------------------------------------------------
        # 4. Causal mask (only materialised at decode to avoid O(N²) OOM)
        # ------------------------------------------------------------------
        phys_past_len = past_kv[0].shape[-2] if past_kv is not None else 0
        attention_mask = None
        if q_len == 1:
            attention_mask = _make_causal_mask(
                bsz, q_len, phys_past_len, query_states.dtype, query_states.device
            )

        # ------------------------------------------------------------------
        # 5. Rotary Positional Embeddings
        # ------------------------------------------------------------------
        kv_seq_len = key_states.shape[-2] + phys_past_len
        cos, sin = self.rotary_emb(
            value_states, seq_len=max(kv_seq_len, position_ids.max().item() + 1)
        )
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states   = apply_rotary_pos_emb_single(key_states,   cos, sin, position_ids)

        # ------------------------------------------------------------------
        # 6. KV Cache Concatenation
        # ------------------------------------------------------------------
        if past_kv is not None:
            key_states   = torch.cat([past_kv[0], key_states],   dim=2)
            value_states = torch.cat([past_kv[1], value_states], dim=2)

        current_kv = (key_states, value_states) if use_cache else None

        if self.layer_idx == 0 and self._dbg_count < 5:
            has_qcache = hasattr(self.kv_cache, 'q_cache_k_quant') and self.kv_cache.q_cache_k_quant is not None
            print(f"[DBG L0 step={self._dbg_count}] pos_ids={position_ids[0,:3].tolist()}... global_tc={self.kv_cache.global_token_counter.item()} phys_past={phys_past_len} has_qcache={has_qcache} concat_k_shape={key_states.shape}", flush=True)
            self._dbg_count += 1

        # ------------------------------------------------------------------
        # 7. Attention logits
        # ------------------------------------------------------------------
        main_logits = compute_main_logits(
            query_states, key_states,
            bsz, self.num_heads, self.num_key_value_heads,
            self.num_key_value_groups, q_len, self.head_dim,
        )

        if attention_mask is not None:
            main_logits = main_logits + attention_mask
        elif q_len > 1:
            apply_prefill_causal_mask(main_logits, q_len, phys_past_len)

        # ------------------------------------------------------------------
        # 8. Softmax + output
        # ------------------------------------------------------------------
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
            attn_output, scores_for_cache, attn_weights_for_output = \
                compute_standard_softmax(
                    query_states, main_logits, value_states,
                    bsz, self.num_heads, self.num_key_value_heads,
                    self.num_key_value_groups, q_len, self.head_dim,
                )

        # ------------------------------------------------------------------
        # 9. Sticky KV Cache Eviction
        # ------------------------------------------------------------------
        evicted_kv = self.kv_cache(
            current_kv,
            scores_for_cache.detach(),
            full_attn_scores=attn_weights_for_output.detach(),
            q_len=q_len,
            q_attn_scores=q_scores_for_cache.detach() if q_scores_for_cache is not None else None,
        )

        # ------------------------------------------------------------------
        # 10. Write evicted KV back into the cache object IN PLACE.
        #
        #     This is the critical step: we write directly into
        #     cache_obj.key_cache[layer_idx] so the tensors persist
        #     across generate() steps.  Works identically for both
        #     DynamicCache and StickyCache.
        # ------------------------------------------------------------------
        if use_cache and evicted_kv is not None and cache_obj is not None:
            global_tc = int(self.kv_cache.global_token_counter.item())
            _write_kv_to_cache(cache_obj, self.layer_idx, evicted_kv, global_tc)
            
            if self.layer_idx == 0 and self._dbg_count <= 5:
                print(f"[DBG L0 writeback step={self._dbg_count - 1}] cache_id={id(cache_obj)} _seen_tokens={getattr(cache_obj, '_seen_tokens', 'N/A')}", flush=True)

        # ------------------------------------------------------------------
        # 11. Output projection
        # ------------------------------------------------------------------
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # HF >= 4.46: LlamaDecoderLayer does `hidden_states, _ = self.self_attn(...)`
        # The cache is updated in-place above — it is NOT returned here.
        return attn_output, (attn_weights_for_output if output_attentions else None)
