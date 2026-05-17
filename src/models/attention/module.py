"""
attention/module.py
-------------------
STICKYLlamaAttention — standard PyTorch SDPA backend.

Pinned to transformers == 4.57.  The only supported cache layout is
DynamicCache with a `.layers` list of CacheLayerMixin objects, which we
replace on first write with `StickyDynamicLayer` so cumulative position
tracking survives eviction.
"""

import torch
from torch import nn
from typing import Optional

from src.models.sticky_kv_logic_cummulative import (
    _make_causal_mask,
    apply_rotary_pos_emb_single,
    STICKYKVCache_LayerWise,
)
from src.models.sticky_cache import StickyDynamicLayer
from .rope import Llama3RotaryEmbedding, init_rope  # noqa: F401
from .ops import (
    compute_main_logits,
    apply_prefill_causal_mask,
    compute_qcache_joint_softmax,
    compute_standard_softmax,
)


def _read_kv_from_cache(cache_obj, layer_idx):
    """Extract (k, v) from a DynamicCache (transformers >= 4.48).

    Returns (past_kv, cache_obj):
      past_kv   = (k_tensor, v_tensor) or None at prefill / empty layer
      cache_obj = the cache object to write back to (mirrors input)
    """
    if cache_obj is None:
        return None, None

    if len(cache_obj.layers) > layer_idx:
        layer = cache_obj.layers[layer_idx]
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is not None and k.numel() > 0:
            return (k, v), cache_obj
    return None, cache_obj


def _write_kv_to_cache(cache_obj, layer_idx, evicted_kv, global_token_counter):
    """Write evicted KV back into the DynamicCache in-place.

    On first write to a layer we replace HF's vanilla `DynamicLayer` with our
    `StickyDynamicLayer`, which tracks `cumulative_length` (true sequence
    position) separately from the physical KV tensor length — same pattern
    HF uses for `DynamicSlidingWindowLayer` / SinkCache.
    """
    if cache_obj is None or evicted_kv is None:
        return

    while len(cache_obj.layers) <= layer_idx:
        cache_obj.layers.append(StickyDynamicLayer())

    layer = cache_obj.layers[layer_idx]
    if not isinstance(layer, StickyDynamicLayer):
        sticky_layer = StickyDynamicLayer()
        cache_obj.layers[layer_idx] = sticky_layer
        layer = sticky_layer

    layer.set_evicted_kv(evicted_kv[0], evicted_kv[1])
    layer.cumulative_length = int(global_token_counter)


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
        self._dbg_count = 0

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

        # Debug counter (layer 0 only)
        if self.layer_idx == 0 and not hasattr(self, '_dbg_count'):
            self._dbg_count = 0

        # ------------------------------------------------------------------
        # 1. Read KV from the DynamicCache (transformers 4.57).
        #    None at prefill, populated DynamicLayer/StickyDynamicLayer afterwards.
        # ------------------------------------------------------------------
        past_kv, cache_obj = _read_kv_from_cache(past_key_values, self.layer_idx)


        # ------------------------------------------------------------------
        # 2. Position IDs — use TRUE global sequence position.
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
        max_pos = int(position_ids.max().item()) if position_ids.numel() > 0 else (q_len - 1)
        cos, sin = self.rotary_emb(
            value_states, seq_len=max(kv_seq_len, max_pos + 1)
        )
        

        if self.layer_idx == 0 and self._dbg_count < 6:
            print(f"[DBG L0 step={self._dbg_count} PRE-ROPE] Q_norm={query_states.norm().item():.4e}, K_norm={key_states.norm().item():.4e}", flush=True)

        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states   = apply_rotary_pos_emb_single(key_states,   cos, sin, position_ids)

        if self.layer_idx == 0 and self._dbg_count < 6:
            print(f"[DBG L0 step={self._dbg_count} POST-ROPE] Q_norm={query_states.norm().item():.4e}, K_norm={key_states.norm().item():.4e}", flush=True)

        # ------------------------------------------------------------------
        # 6. KV Cache Concatenation
        # ------------------------------------------------------------------
        step1_phys_before = past_kv[0].shape[-2] if past_kv is not None else 0
        if past_kv is not None:
            key_states   = torch.cat([past_kv[0], key_states],   dim=2)
            value_states = torch.cat([past_kv[1], value_states], dim=2)

        trace_decode = False
        if self.layer_idx == 0 and q_len == 1:
            em = getattr(self.kv_cache, "eviction_manager", None)
            tokens_since = getattr(em, "tokens_since_last_review", None)
            omega = getattr(self.kv_cache, "omega", None)
            trace_decode = (
                self._dbg_count <= 10
                or (omega is not None and tokens_since in {max(int(omega) - 1, 0), int(omega)})
            )

        if self.layer_idx == 0 and self._dbg_count == 1:
            cache_type = type(past_key_values).__name__ if past_key_values is not None else "None"
            pos_dbg = position_ids.detach().flatten().tolist()
            global_tc = int(self.kv_cache.global_token_counter.item())
            print(
                f"[STEP1-CACHE current] cache_type={cache_type} "
                f"q_len={q_len} pos_ids={pos_dbg} global_tc={global_tc} "
                f"phys_before={step1_phys_before} concat_len={key_states.shape[-2]} "
                f"use_cache={use_cache}",
                flush=True,
            )

        if trace_decode:
            cache_type = type(past_key_values).__name__ if past_key_values is not None else "None"
            pos_dbg = position_ids.detach().flatten().tolist()
            global_tc = int(self.kv_cache.global_token_counter.item())
            em = getattr(self.kv_cache, "eviction_manager", None)
            tokens_since = getattr(em, "tokens_since_last_review", "NA")
            gen_step = getattr(em, "gen_step", "NA")
            q_cache_ids = getattr(self.kv_cache, "q_cache_ids", None)
            q_windows = q_cache_ids.shape[1] if q_cache_ids is not None else 0
            qcache_active = (
                hasattr(self.kv_cache, "q_cache_k_quant")
                and self.kv_cache.q_cache_k_quant is not None
            )
            cache_position = kwargs.get("cache_position", None)
            cache_position_dbg = (
                cache_position.detach().flatten().tolist()
                if torch.is_tensor(cache_position)
                else cache_position
            )
            print(
                f"[DECODE-TRACE current PRE step={self._dbg_count}] "
                f"cache_type={cache_type} q_len={q_len} pos_ids={pos_dbg} global_tc={global_tc} "
                f"phys_before={step1_phys_before} concat_len={key_states.shape[-2]} "
                f"tokens_since={tokens_since} gen_step={gen_step} q_windows={q_windows} "
                f"qcache_active={qcache_active} "
                f"cache_position={cache_position_dbg} use_cache={use_cache}",
                flush=True,
            )

        current_kv = (key_states, value_states) if use_cache else None

        # ------------------------------------------------------------------
        # 7. Attention logits
        # ------------------------------------------------------------------
        main_logits = compute_main_logits(
            query_states, key_states,
            bsz, self.num_heads, self.num_key_value_heads,
            self.num_key_value_groups, q_len, self.head_dim,
        )

        if self.layer_idx == 0 and self._dbg_count < 6:
            q_w_n = self.q_proj.weight.norm().item()
            k_w_n = self.k_proj.weight.norm().item()
            q_n = query_states.norm().item()
            k_n = key_states.norm().item()
            m_n = main_logits.norm().item()
            print(f"[DBG L0 step={self._dbg_count} LOGITS] q_proj_W={q_w_n:.4e}, k_proj_W={k_w_n:.4e}, Q_norm={q_n:.4e}, K_norm={k_n:.4e}, logits_norm={m_n:.4e}. Max val={main_logits.max().item():.4e}, Min val={main_logits.min().item():.4e}", flush=True)

        if trace_decode:
            print(
                f"[DECODE-TRACE current LOGITS step={self._dbg_count}] "
                f"main_shape={tuple(main_logits.shape)} logits_norm={main_logits.norm().item():.4e} "
                f"max={main_logits.max().item():.4e} min={main_logits.min().item():.4e}",
                flush=True,
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

        if trace_decode:
            scores_shape = tuple(scores_for_cache.shape) if scores_for_cache is not None else None
            q_scores_shape = tuple(q_scores_for_cache.shape) if q_scores_for_cache is not None else None
            out_scores_shape = tuple(attn_weights_for_output.shape) if attn_weights_for_output is not None else None
            print(
                f"[DECODE-TRACE current SCORES step={self._dbg_count}] "
                f"scores_shape={scores_shape} q_scores_shape={q_scores_shape} "
                f"output_attn_shape={out_scores_shape}",
                flush=True,
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

        if trace_decode:
            em = getattr(self.kv_cache, "eviction_manager", None)
            tokens_since = getattr(em, "tokens_since_last_review", "NA")
            gen_step = getattr(em, "gen_step", "NA")
            q_cache_ids = getattr(self.kv_cache, "q_cache_ids", None)
            q_windows = q_cache_ids.shape[1] if q_cache_ids is not None else 0
            evicted_len = evicted_kv[0].shape[-2] if evicted_kv is not None else "None"
            global_tc = int(self.kv_cache.global_token_counter.item())
            print(
                f"[DECODE-TRACE current POST step={self._dbg_count}] "
                f"global_tc={global_tc} evicted_len={evicted_len} "
                f"tokens_since={tokens_since} gen_step={gen_step} q_windows={q_windows}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # 10. Write evicted KV back into the StickyDynamicLayer in-place
        #     so the tensors persist across generate() steps.
        # ------------------------------------------------------------------
        if use_cache and evicted_kv is not None and cache_obj is not None:
            global_tc = int(self.kv_cache.global_token_counter.item())
            _write_kv_to_cache(cache_obj, self.layer_idx, evicted_kv, global_tc)
            if trace_decode:
                cache_seq_len = cache_obj.get_seq_length()
                print(
                    f"[DECODE-TRACE current WRITE step={self._dbg_count}] "
                    f"cache_seq_len={cache_seq_len} written_global_tc={global_tc}",
                    flush=True,
                )

        # ------------------------------------------------------------------
        # 11. Output projection
        # ------------------------------------------------------------------
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if self.layer_idx == 0 and hasattr(self, '_dbg_count'):
            self._dbg_count += 1  # Increment AFTER all debug prints for this step

        # transformers 4.57: LlamaDecoderLayer does `hidden_states, _ = self.self_attn(...)`
        # The cache is updated in-place above — it is NOT returned here.
        return attn_output, (attn_weights_for_output if output_attentions else None)
