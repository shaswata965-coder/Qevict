"""
attention/module_flash.py
-------------------------
STICKYLlamaAttention — Flash-Attention v2 backend.

Pinned to transformers == 4.57. Same cache contract as module.py
(DynamicCache with StickyDynamicLayer + global_token_counter).
Prefill uses Flash-Attention + chunked tracking scores (ops_flash.py).
Decoding reuses the shared SDPA ops (ops.py).
"""

import torch
from torch import nn
from typing import Optional, Tuple

from src.models.sticky_kv_logic_fast_attention import (
    _make_causal_mask,
    apply_rotary_pos_emb_single,
    STICKYKVCache_LayerWise,
)
from .module import _read_kv_from_cache, _write_kv_to_cache
from .rope import Llama3RotaryEmbedding, init_rope  # noqa: F401
from .ops import (
    compute_main_logits,
    compute_qcache_joint_softmax,
    compute_standard_softmax,
)
from .ops_flash import prefill_flash_attention, compute_chunked_prefill_scores


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
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # ------------------------------------------------------------------
        # 1. Extract (k, v) tuple from the DynamicCache (transformers 4.57)
        # ------------------------------------------------------------------
        past_kv, cache_obj = _read_kv_from_cache(past_key_values, self.layer_idx)

        # Debug counter (layer 0 only)
        if self.layer_idx == 0 and not hasattr(self, '_dbg_count'):
            self._dbg_count = 0

        if self.layer_idx == 0 and self._dbg_count < 5:
            cache_type = type(past_key_values).__name__ if past_key_values is not None else 'None'
            cache_id = id(past_key_values) if past_key_values is not None else 0
            past_kv_shape = past_kv[0].shape if past_kv is not None else 'None'
            print(f"[DBG L0 step={self._dbg_count}] q_len={q_len} cache_type={cache_type} (id={cache_id}) past_kv_shape={past_kv_shape}", flush=True)

        # ------------------------------------------------------------------
        # 2. Position IDs — TRUE global sequence position
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
        # 4. Causal mask
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

        if q_len > 1:
            # ----------------------------------------------------------
            # PREFILL — Flash-Attention v2
            # ----------------------------------------------------------
            attn_output = prefill_flash_attention(
                query_states, key_states, value_states, self.head_dim
            )
            attn_weights_return = None

            accumulated_scores = compute_chunked_prefill_scores(
                query_states, key_states,
                q_len, phys_past_len,
                self.num_key_value_heads, self.num_heads, self.num_key_value_groups,
                self.head_dim,
            )

            evicted_kv = self.kv_cache(
                current_kv, accumulated_scores.detach(), q_len=q_len
            )

        else:
            # ----------------------------------------------------------
            # DECODE — Standard SDPA (identical to SDPA backend)
            # ----------------------------------------------------------
            main_logits = compute_main_logits(
                query_states, key_states,
                bsz, self.num_heads, self.num_key_value_heads,
                self.num_key_value_groups, q_len, self.head_dim,
            )

            if attention_mask is not None:
                main_logits = main_logits + attention_mask

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

            evicted_kv = self.kv_cache(
                current_kv,
                scores_for_cache.detach(),
                q_attn_scores=q_scores_for_cache.detach() if q_scores_for_cache is not None else None,
                q_len=q_len,
            )

            attn_weights_return = attn_weights_for_output if output_attentions else None

        # ------------------------------------------------------------------
        # 7. Write evicted KV back into the cache object in-place
        # ------------------------------------------------------------------
        if use_cache and evicted_kv is not None and cache_obj is not None:
            global_tc = int(self.kv_cache.global_token_counter.item())
            _write_kv_to_cache(cache_obj, self.layer_idx, evicted_kv, global_tc)

            if self.layer_idx == 0 and self._dbg_count <= 5:
                print(f"[DBG L0 writeback step={self._dbg_count - 1}] cache_id={id(cache_obj)} cumulative_length={cache_obj.layers[self.layer_idx].cumulative_length}", flush=True)

        # ------------------------------------------------------------------
        # 8. Output projection
        # ------------------------------------------------------------------
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights_return
