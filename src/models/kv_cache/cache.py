"""
kv_cache/cache.py
-----------------
STICKYKVCache_LayerWise — lean coordinator for the Sticky KV-cache.

All eviction logic  →  EvictionManager    (eviction_manager.py)
All quantization    →  QuantizationManager (quantize_manager.py)
All tracking/ledger →  TrackingManager     (tracking_manager.py)
Budget allocation   →  compute_budget()    (allocator.py)
Pure KV helpers     →  eviction.py, quantize.py, helpers.py, ledger.py

External API is UNCHANGED:
  - STICKYKVCache_LayerWise(p_ratio, r_ratio, start_idx, num_heads, layer_idx, config)
  - cache(past_key_values, attn_score_cache, full_attn_scores, q_len, q_attn_scores)
  - cache._clean_scores()
  - cache.get_ledger_data()
  - cache.q_cache_k_quant / q_cache_k_scale / q_cache_k_zp  (delegating properties)
  - cache.q_cache_v_quant / q_cache_v_scale / q_cache_v_zp  (delegating properties)
  - cache.token_ledger / cache.global_token_counter           (delegating properties)
  - cache._dequantize_from_quant(...)                         (@staticmethod)
"""

import torch
from torch import nn

from .helpers import repeat_kv, _make_causal_mask, apply_rotary_pos_emb_single  # noqa: F401
from .quantize import quantize_k_per_window, quantize_v_per_window, dequantize_from_quant
from .allocator import compute_budget
from .eviction import (
    find_logical_window_span,
    gather_window_from_current_kv,
    evict_from_window_scores,
    create_mask_and_evict_from_kv_cache_prompt_stage,
)
from .eviction_manager import EvictionManager, DecodeEvictionResult
from .quantize_manager import QuantizationManager, NoOpQuantizationManager
from .tracking_manager import TrackingManager, NoOpTrackingManager
from .rerotation import rerotate_keys  # kept for potential external callers; not used in cache.py


class STICKYKVCache_LayerWise(nn.Module):
    """Coordinator: delegates eviction, quantization, and tracking to managers."""

    def __init__(
        self,
        p_ratio,
        r_ratio,
        start_idx,
        num_heads,
        layer_idx,
        config=None,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        super().__init__()

        # --- Config scalars ---
        self.total_cache_ratio = r_ratio
        self.local_cache_ratio = p_ratio
        self.k_windows = 3
        self.start_idx = start_idx
        self.k_seq_dim, self.v_seq_dim = k_seq_dim, v_seq_dim
        self.layer_idx = layer_idx
        self.num_heads = num_heads

        try:
            from src.sticky_config import OMEGA, SINK_TOKENS
        except ImportError:
            from sticky_config import OMEGA, SINK_TOKENS  # fallback: running from inside src/
        self.omega = OMEGA
        self.sink_tokens = SINK_TOKENS

        try:
            from src.sticky_config import Q_RATIO
        except ImportError:
            from sticky_config import Q_RATIO  # fallback
        self.q_ratio = Q_RATIO

        try:
            from src.sticky_config import QUANTIZATION_BIT_WIDTH
        except ImportError:
            from sticky_config import QUANTIZATION_BIT_WIDTH  # fallback
        self.quant_bit_width = QUANTIZATION_BIT_WIDTH

        try:
            from src.sticky_config import P_RATIO
            self.local_cache_ratio = P_RATIO
        except ImportError:
            try:
                from sticky_config import P_RATIO
                self.local_cache_ratio = P_RATIO
            except ImportError:
                pass  # keep p_ratio passed as argument

        try:
            from src.sticky_config import LOCAL_NUM_TOKENS
            self.local_num_tokens = LOCAL_NUM_TOKENS
            self.use_fixed_local_tokens = True
        except ImportError:
            try:
                from sticky_config import LOCAL_NUM_TOKENS
                self.local_num_tokens = LOCAL_NUM_TOKENS
                self.use_fixed_local_tokens = True
            except ImportError:
                if config is not None and hasattr(config, "local_num_tokens"):
                    self.local_num_tokens = config.local_num_tokens
                    self.use_fixed_local_tokens = True
                else:
                    self.local_num_tokens = 0
                    self.use_fixed_local_tokens = False

        try:
            from src.sticky_config import tracking_flag
        except ImportError:
            try:
                from sticky_config import tracking_flag  # fallback
            except ImportError:
                tracking_flag = getattr(config, "tracking_flag", 1)
        self._tracking_flag = (tracking_flag == 1)

        if config is not None and hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            self.head_dim = config.hidden_size // config.num_attention_heads
        else:
            self.head_dim = 64

        # Budget counters (updated at prefill by compute_budget)
        self.local_num = 0
        self.q_windows_count = 0
        self.q_num = 0
        self._quant_bytes_len = self.head_dim if self.quant_bit_width == 8 else (self.head_dim // 2)

        # Lifecycle state
        self._prefill_done = False
        self.gen_step = 0
        self.cache_size = int(
            self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens
        )

        # --- Index buffers (registered so .to(device) moves them) ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config is not None and hasattr(config, "max_position_embeddings"):
            max_context = config.max_position_embeddings
            max_windows = max((max_context - self.sink_tokens) // self.omega, 1) if max_context > self.sink_tokens else 1
            max_windows = max(max_windows, 100)
        else:
            max_context = 131072
            max_windows = 30000

        window_ids = torch.arange(max_windows, device=device)
        token_map = (window_ids.unsqueeze(1) * self.omega + self.sink_tokens) + torch.arange(self.omega, device=device)
        self.register_buffer("window_to_token_map", token_map)
        self.register_buffer(
            "sink_indices",
            torch.arange(0, self.sink_tokens, device=device) if self.sink_tokens > 0
            else torch.zeros(0, dtype=torch.long, device=device),
        )

        # --- Instantiate managers ---
        self.eviction_manager = EvictionManager(
            num_heads=num_heads,
            max_windows=max_windows,
            max_context=max_context,
            omega=self.omega,
            sink_tokens=self.sink_tokens,
            window_to_token_map=self.window_to_token_map,
            sink_indices=self.sink_indices,
            device=device,
        )

        self.quantization_manager = (
            QuantizationManager(self.quant_bit_width, self.head_dim, self.omega, num_heads, self.sink_tokens)
            if self.q_ratio > 0
            else NoOpQuantizationManager()
        )

        self.tracking_manager = (
            TrackingManager(max_context, num_heads, device)
            if self._tracking_flag
            else NoOpTrackingManager(device)
        )

        self.rotary_emb = None

    def set_rotary_emb(self, rotary_emb: nn.Module) -> None:
        """Called by STICKYLlamaAttention.__init__ to provide RoPE access."""
        self.rotary_emb = rotary_emb
        self.quantization_manager.set_rotary_emb(rotary_emb)

    # ------------------------------------------------------------------
    # Delegating properties — preserve external API (ops.py, module.py,
    # module_flash.py, eval scripts)
    # ------------------------------------------------------------------

    @property
    def q_cache_k_quant(self):  return self.quantization_manager.q_cache_k_quant
    @property
    def q_cache_k_scale(self):  return self.quantization_manager.q_cache_k_scale
    @property
    def q_cache_k_zp(self):     return self.quantization_manager.q_cache_k_zp
    @property
    def q_cache_v_quant(self):  return self.quantization_manager.q_cache_v_quant
    @property
    def q_cache_v_scale(self):  return self.quantization_manager.q_cache_v_scale
    @property
    def q_cache_v_zp(self):     return self.quantization_manager.q_cache_v_zp
    @property
    def q_cache_ids(self):      return self.quantization_manager.q_cache_ids
    @property
    def q_cache_scores(self):   return self.quantization_manager.q_cache_scores
    @property
    def token_ledger(self):             return self.tracking_manager.token_ledger
    @property
    def global_token_counter(self):     return self.tracking_manager.global_token_counter
    # --- Eviction-manager state exposed for eval scripts ---
    @property
    def window_scores(self):            return self.eviction_manager.window_scores
    # --- Tracking-manager state exposed for eval scripts ---
    @property
    def prefill_attention_matrix(self): return self.tracking_manager.prefill_attention_matrix
    @property
    def global_score_history(self):     return self.tracking_manager._global_score_history

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        past_key_values,
        attn_score_cache,
        full_attn_scores=None,
        q_len=None,
        q_attn_scores=None,
    ):
        device = attn_score_cache.device
        if getattr(self, '_current_device', None) != device:
            if hasattr(self.eviction_manager, 'to'): self.eviction_manager.to(device)
            if hasattr(self.quantization_manager, 'to'): self.quantization_manager.to(device)
            if hasattr(self.tracking_manager, 'to'): self.tracking_manager.to(device)
            self._current_device = device

        _, _, q_len_cache, _ = attn_score_cache.shape
        q_len = q_len if q_len is not None else q_len_cache

        # --- Global token counter ---
        global_start = 0
        if not self._prefill_done:
            self.tracking_manager._global_token_counter.zero_()
            global_start = 0
            self.tracking_manager.increment(n=q_len)
        else:
            global_start = int(self.tracking_manager.global_token_counter.item())
            self.tracking_manager.increment(n=1)

        if past_key_values is None:
            return past_key_values
        seq_len = past_key_values[0].size(self.k_seq_dim)

        # ================================================================
        # PREFILL STAGE
        # ================================================================
        if not self._prefill_done:
            try:
                import src.sticky_config as config_module
            except ImportError:
                import sticky_config as config_module  # fallback: running from inside src/
            budget = compute_budget(
                total_cache_ratio=self.total_cache_ratio,
                sink_tokens=self.sink_tokens,
                use_fixed_local_tokens=self.use_fixed_local_tokens,
                local_num_tokens=self.local_num_tokens,
                local_cache_ratio=self.local_cache_ratio,
                q_ratio=self.q_ratio,
                quant_bit_width=self.quant_bit_width,
                head_dim=self.head_dim,
                omega=self.omega,
                new_tokens=q_len,
                max_tokens=config_module.GENERATION_CONFIG.get("max_new_tokens", 512),
                layer_idx=self.layer_idx,
            )
            self.local_num = budget.local_num
            self.k_windows = budget.k_windows
            self.q_windows_count = budget.q_windows_count
            self.q_num = budget.q_num
            self._quant_bytes_len = budget.quant_bytes_len
            self.cache_size = (
                self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens
            )

            # Parity/debug mode: R_RATIO=100 must mean the prefill KV cache is
            # left byte-for-byte intact.  The decode path already bypasses
            # physical eviction at 100%, but without this guard prefill still
            # rebuilt the cache through the sticky-window machinery, which made
            # "no eviction" runs diverge before the first generated token.
            if self.total_cache_ratio >= 100 and self.q_ratio == 0:
                self._prefill_done = True
                return past_key_values

            self.eviction_manager.num_of_tokens_without_eviction += seq_len
            for h in range(self.num_heads):
                self.eviction_manager.prompt_boundary[h] = seq_len - 1

            # Tally scores → get score_end for physical eviction
            score_end = self.eviction_manager.tally_prefill_scores(
                attn_score_cache, seq_len, self.local_num
            )

            # Eviction selection
            evict_result = self.eviction_manager.get_prefill_eviction(
                self.k_windows, self.q_windows_count
            )

            # Quantise loser windows into Q-cache
            if evict_result.loser_ids is not None:
                q_count = evict_result.loser_ids.shape[1]
                q_phys = self.window_to_token_map[evict_result.loser_ids.long()]
                q_phys_flat = q_phys.reshape(self.num_heads, -1).clamp(0, seq_len - 1)
                hd = past_key_values[0].shape[-1]
                gather_q = q_phys_flat.unsqueeze(-1).expand(-1, -1, hd)
                q_k = torch.gather(past_key_values[0][0], 1, gather_q).view(self.num_heads, q_count, self.omega, hd)
                q_v = torch.gather(past_key_values[1][0], 1, gather_q).view(self.num_heads, q_count, self.omega, hd)
                self.quantization_manager.store_windows(
                    q_k, q_v, evict_result.loser_ids, evict_result.loser_scores, q_phys
                )

            # Physical eviction
            updated_kv, survivor_ids = create_mask_and_evict_from_kv_cache_prompt_stage(
                past_key_values, attn_score_cache, score_end,
                k_seq_dim=self.k_seq_dim,
                window_scores=self.eviction_manager.window_scores,
                k_windows=self.k_windows,
                sink_indices=self.sink_indices,
                window_to_token_map=self.window_to_token_map,
                num_heads=self.num_heads,
            )

            self.eviction_manager.commit_prefill_result(survivor_ids)

            # Tracking
            self.tracking_manager.record_prefill(
                seq_len=seq_len, q_len=q_len, layer_idx=self.layer_idx,
                attn_score_cache=attn_score_cache, survivor_ids=survivor_ids,
                full_attn_scores=full_attn_scores, global_start=global_start,
                num_new_tokens=q_len,
            )

            self._prefill_done = True
            return updated_kv

        # ================================================================
        # DECODE STAGE
        # ================================================================
        em = self.eviction_manager
        device = em.window_scores.device

        if em.logical_id_map is None:
            return past_key_values

        em.num_of_tokens_without_eviction += 1
        em.gen_step += 1

        em.tally_decode_scores(attn_score_cache, seq_len)
        self.quantization_manager.accumulate_scores(q_attn_scores, self.omega)
        self.tracking_manager.record_decode(
            attn_score_cache, seq_len, self.layer_idx, global_start
        )

        if em.tokens_since_last_review != self.omega:
            return past_key_values

        # --- Eviction cycle ---
        result = em.run_decode_cycle(
            k_windows=self.k_windows,
            q_windows_count=self.q_windows_count,
            local_tokens_count=em._dynamic_local_count,
            q_cache_ids=self.quantization_manager.q_cache_ids,
            q_cache_scores=self.quantization_manager.q_cache_scores,
            seq_len=seq_len,
        )

        em.update_local_history(result)

        # Early-exit if no physical eviction needed
        if self.total_cache_ratio == 100:
            em.reset_decode_counters(seq_len)
            return past_key_values

        # Q-cache promotions
        promoted_k, promoted_v = self.quantization_manager.get_promoted_windows(result.winner_ids)

        # Q-cache rebuild
        self.quantization_manager.rebuild(
            new_loser_ids=result.loser_ids,
            new_loser_scores=result.loser_scores,
            past_key_values=past_key_values,
            pre_block_wids=result.pre_block_wids,
            seq_len=seq_len,
            omega=self.omega,
            layer_idx=self.layer_idx,
        )

        # Physical cache rebuild
        updated_kv, new_lid_map, mapping = self._build_physical_cache(
            past_key_values, result, promoted_k, promoted_v,
            em._dynamic_local_count, device,
        )

        em.commit_decode_result(result, new_lid_map)

        # Tracking position update
        self.tracking_manager.update_physical_positions(
            mapping, seq_len, em.running_attention_votes
        )

        em.reset_decode_counters(seq_len)
        return updated_kv

    # ------------------------------------------------------------------
    # Private: physical cache reconstruction
    # ------------------------------------------------------------------

    def _build_physical_cache(self, past_key_values, result: DecodeEvictionResult,
                               promoted_k, promoted_v, local_tokens_count, device):
        """Reconstruct the dense BF16 KV cache after a decode eviction cycle.

        Returns (updated_kv, new_logical_id_map, mapping).
        mapping is [H, seq_len] float32 mapping old→new positions (-1=evicted).
        """
        em = self.eviction_manager
        omega = self.omega
        sink_tokens = self.sink_tokens
        num_heads = self.num_heads
        curr_k = result.curr_k
        final_ids = result.winner_ids
        seq_len = past_key_values[0].shape[self.k_seq_dim]

        head_dim = past_key_values[0].shape[-1]
        dtype_fp = past_key_values[0].dtype

        new_compressed_len = sink_tokens + curr_k * omega
        new_seq_len = new_compressed_len + local_tokens_count
        new_k = torch.zeros(1, num_heads, new_seq_len, head_dim, device=device, dtype=dtype_fp)
        new_v = torch.zeros(1, num_heads, new_seq_len, head_dim, device=device, dtype=dtype_fp)
        new_lid_map = torch.full((num_heads, new_seq_len), -1, device=device, dtype=torch.long)
        mapping = torch.full((num_heads, seq_len), -1.0, device=device, dtype=torch.float32) if self._tracking_flag else None

        num_old_blocks = result.pre_num_old_blocks
        if num_old_blocks > 0:
            block_wids = result.pre_block_wids
            match = (block_wids.unsqueeze(1) == final_ids.unsqueeze(2))
            found_in_main = match.any(dim=2)
            slot_idx = match.to(torch.uint8).argmax(dim=2)
            first_phys = sink_tokens + slot_idx * omega
            # Guard: exclude windows whose physical span exceeds the old cache
            in_bounds = (first_phys + omega) <= seq_len
            found_in_main = found_in_main & in_bounds
        else:
            found_in_main = torch.zeros(num_heads, curr_k, device=device, dtype=torch.bool)
            first_phys = torch.zeros(num_heads, curr_k, device=device, dtype=torch.long)

        _prom_k = {(h, int(w)): k for h in range(num_heads) for w, k in promoted_k[h]}
        _prom_v = {(h, int(w)): v for h in range(num_heads) for w, v in promoted_v[h]}

        _local_lids = None
        if local_tokens_count > 0:
            offsets = torch.arange(local_tokens_count, device=device, dtype=torch.long)
            local_start_wid = max(
                0, (em.num_of_tokens_without_eviction - sink_tokens - local_tokens_count) // omega
            )
            _local_lids = local_start_wid + (offsets // omega)

        # 1. Sinks
        new_k[0, :, :sink_tokens] = past_key_values[0][0, :, :sink_tokens]
        new_v[0, :, :sink_tokens] = past_key_values[1][0, :, :sink_tokens]
        new_lid_map[:, :sink_tokens] = em.logical_id_map[:, :sink_tokens]
        if mapping is not None:
            mapping[:, :sink_tokens] = torch.arange(sink_tokens, device=device, dtype=torch.float32).unsqueeze(0)

        # 2. Sticky windows — vectorised batch copy
        if found_in_main.any():
            target_starts = (sink_tokens + torch.arange(curr_k, device=device, dtype=torch.long) * omega).unsqueeze(0).expand(num_heads, -1)
            offsets_om = torch.arange(omega, device=device, dtype=torch.long)
            phys_gather  = (first_phys.unsqueeze(2) + offsets_om).view(num_heads, -1)
            target_scatter = (target_starts.unsqueeze(2) + offsets_om).view(num_heads, -1)
            mask = found_in_main.unsqueeze(2).expand(-1, -1, omega).reshape(num_heads, -1)
            valid_phys   = phys_gather[mask]
            valid_target = target_scatter[mask]
            h_exp = torch.arange(num_heads, device=device).unsqueeze(1).expand(-1, curr_k * omega)
            valid_heads = h_exp[mask]
            new_k[0, valid_heads, valid_target] = past_key_values[0][0, valid_heads, valid_phys]
            new_v[0, valid_heads, valid_target] = past_key_values[1][0, valid_heads, valid_phys]
            flat_final_ids = final_ids.unsqueeze(2).expand(-1, -1, omega).reshape(num_heads, -1)
            new_lid_map[valid_heads, valid_target] = flat_final_ids[mask].long()
            if mapping is not None:
                mapping[valid_heads, valid_phys] = valid_target.to(mapping.dtype)

        # 3. Non-main windows (promoted from Q-cache or fallback via logical_id_map)
        not_in_main_mask = ~found_in_main
        if not_in_main_mask.any():
            heads, indices = not_in_main_mask.nonzero(as_tuple=True)
            for h_idx, i_idx, wid_val in zip(heads.tolist(), indices.tolist(), final_ids[heads, indices].long().tolist()):
                new_pos = sink_tokens + i_idx * omega
                p_k = _prom_k.get((h_idx, wid_val))
                p_v = _prom_v.get((h_idx, wid_val))
                if p_k is not None:
                    new_k[0, h_idx, new_pos:new_pos + omega] = p_k
                    new_v[0, h_idx, new_pos:new_pos + omega] = p_v
                    new_lid_map[h_idx, new_pos:new_pos + omega] = wid_val
                    if mapping is not None:
                        span = find_logical_window_span(em.logical_id_map, omega, h_idx, wid_val, seq_len)
                        if span is not None:
                            old_s, old_e = span
                            mapping[h_idx, old_s:old_e] = torch.arange(
                                new_pos, new_pos + omega, device=device, dtype=torch.float32
                            )
                else:
                    span = find_logical_window_span(em.logical_id_map, omega, h_idx, wid_val, seq_len)
                    if span is not None:
                        old_s, old_e = span
                        new_k[0, h_idx, new_pos:new_pos + omega] = past_key_values[0][0, h_idx, old_s:old_e]
                        new_v[0, h_idx, new_pos:new_pos + omega] = past_key_values[1][0, h_idx, old_s:old_e]
                        new_lid_map[h_idx, new_pos:new_pos + omega] = wid_val
                        if mapping is not None:
                            mapping[h_idx, old_s:old_e] = torch.arange(new_pos, new_pos + omega, device=device, dtype=torch.float32)
                    else:
                        new_lid_map[h_idx, new_pos:new_pos + omega] = wid_val

        # 4. Local zone
        if local_tokens_count > 0:
            old_local_start = seq_len - local_tokens_count
            new_local_start = new_compressed_len
            actual_local = min(local_tokens_count, seq_len - old_local_start)
            new_k[0, :, new_local_start:new_local_start + actual_local] = past_key_values[0][0, :, old_local_start:old_local_start + actual_local]
            new_v[0, :, new_local_start:new_local_start + actual_local] = past_key_values[1][0, :, old_local_start:old_local_start + actual_local]
            if _local_lids is not None:
                new_lid_map[:, new_local_start:new_local_start + actual_local] = _local_lids[:actual_local].unsqueeze(0)
            if mapping is not None:
                mapping[:, old_local_start:old_local_start + actual_local] = torch.arange(
                    new_local_start, new_local_start + actual_local, device=device, dtype=torch.float32
                ).unsqueeze(0)

        return (new_k, new_v), new_lid_map, mapping

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _clean_scores(self):
        """Reset all cache state between documents / sequences."""
        self.k_windows = 3
        self.local_num = 0
        self.q_windows_count = 0
        self.q_num = 0
        self._prefill_done = False
        self.gen_step = 0
        self.cache_size = int(
            self.omega * (1 + self.local_num + self.k_windows + self.start_idx) + self.sink_tokens
        )
        self.eviction_manager.reset()
        self.quantization_manager.reset()
        self.tracking_manager.reset()
        self._quant_bytes_len = self.head_dim if self.quant_bit_width == 8 else (self.head_dim // 2)

    # ------------------------------------------------------------------
    # Accessor helpers (preserve original method signatures)
    # ------------------------------------------------------------------

    def get_ledger_data(self):
        return self.tracking_manager.get_ledger_data()

    def _find_logical_window_span(self, h, wid_val, seq_len):
        return find_logical_window_span(
            self.eviction_manager.logical_id_map, self.omega, h, wid_val, seq_len
        )

    def _gather_window_from_current_kv(self, past_key_values, h, wid_val, *, seq_len):
        return gather_window_from_current_kv(
            self.eviction_manager.logical_id_map, self.omega,
            past_key_values, h, wid_val, seq_len=seq_len,
        )

    def _update_k_win_and_local_num(self, new_tokens, max_tokens):
        """Deprecated shim — budget is now computed via compute_budget() at prefill."""
        from .allocator import update_k_win_and_local_num
        update_k_win_and_local_num(self, new_tokens, max_tokens)

    # ------------------------------------------------------------------
    # Static method — called by FA2 ops.py externally
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_k_per_window(tensor, bit_width=8):
        return quantize_k_per_window(tensor, bit_width)

    @staticmethod
    def _quantize_v_per_window(tensor, bit_width=8):
        return quantize_v_per_window(tensor, bit_width)

    @staticmethod
    def _dequantize_from_quant(quant_tensor, scale, zero_point, bit_width=8):
        return dequantize_from_quant(quant_tensor, scale, zero_point, bit_width)
