"""
attention/module.py
-------------------
STICKYLlamaAttention — standard PyTorch SDPA backend.

Key design:
  - Uses StickyCache (a Cache subclass) to carry evicted KV tuples through
    HF's generate loop, replacing the old DynamicCache in-place mutation.
  - position_ids during decode are derived from global_token_counter (the
    TRUE sequence position), NOT from the compressed physical cache size.
    This is the critical fix that matches the original implementation.
"""

import torch
from torch import nn
from typing import Optional, Tuple

from src.models.sticky_kv_logic_cummulative import (
    _make_causal_mask,
    apply_rotary_pos_emb_single,
    STICKYKVCache_LayerWise,
)
try:
    from src.models.sticky_cache import StickyCache
except ImportError:
    from ..sticky_cache import StickyCache
from .rope import Llama3RotaryEmbedding, init_rope  # noqa: F401
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

        # ------------------------------------------------------------------
        # 1. Extract (k, v) tuple from whatever cache type HF passed.
        #
        #   StickyCache   → read from typed slot       (primary path)
        #   DynamicCache  → read from key_cache list   (fallback)
        #   tuple         → use directly               (very old HF / tests)
        #   None          → prefill, no prior context
        # ------------------------------------------------------------------
        sticky_cache = None
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if isinstance(past_key_value, StickyCache):
            sticky_cache = past_key_value
            past_kv = sticky_cache.get_kv(self.layer_idx)
        elif isinstance(past_key_value, tuple):
            past_kv = past_key_value
        elif past_key_value is not None and hasattr(past_key_value, "key_cache"):
            sticky_cache = past_key_value
            if len(past_key_value.key_cache) > self.layer_idx:
                k = past_key_value.key_cache[self.layer_idx]
                v = past_key_value.value_cache[self.layer_idx]
                if k.numel() > 0:
                    past_kv = (k, v)

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
        # 10. Write evicted KV back into the cache object
        # ------------------------------------------------------------------
        if use_cache and evicted_kv is not None:
            if isinstance(sticky_cache, StickyCache):
                sticky_cache.set_kv(self.layer_idx, evicted_kv)
            elif sticky_cache is not None and hasattr(sticky_cache, "key_cache"):
                while len(sticky_cache.key_cache) <= self.layer_idx:
                    sticky_cache.key_cache.append(torch.empty(0))
                    sticky_cache.value_cache.append(torch.empty(0))
                sticky_cache.key_cache[self.layer_idx]   = evicted_kv[0]
                sticky_cache.value_cache[self.layer_idx] = evicted_kv[1]

        # ------------------------------------------------------------------
        # 11. Output projection
        # ------------------------------------------------------------------
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Modern HF (≥ 4.46): LlamaDecoderLayer does `hidden_states, _ = self.self_attn(...)`
        return attn_output, (attn_weights_for_output if output_attentions else None)
