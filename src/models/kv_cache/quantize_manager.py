"""
kv_cache/quantize_manager.py
-----------------------------
Concrete implementations of BaseQuantizationManager:

  QuantizationManager     — active path (Q_RATIO > 0)
  NoOpQuantizationManager — inactive path (Q_RATIO == 0)

All three Q-cache rebuild routing paths are implemented here:
  Path A — retained from previous Q-cache (raw int copy)
  Path B — fresh from main BF16 cache (batch gather + quantise)
  Path C — archived meta available (re-quantise with stored scale/zp)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .base_quantization_manager import BaseQuantizationManager
from .quantize import quantize_k_per_window, quantize_v_per_window, dequantize_from_quant


# ---------------------------------------------------------------------------
# Active implementation
# ---------------------------------------------------------------------------

class QuantizationManager(BaseQuantizationManager):
    """Manages the quantised eviction side-cache (Q-cache)."""

    def __init__(self, quant_bit_width: int, head_dim: int, omega: int, num_heads: int, sink_tokens: int = 0):
        self.quant_bit_width = quant_bit_width
        self.head_dim = head_dim
        self.omega = omega
        self.num_heads = num_heads
        self.sink_tokens = sink_tokens
        self._quant_bytes_len = head_dim if quant_bit_width == 8 else (head_dim // 2)

        self._q_cache_k_quant: Optional[torch.Tensor] = None
        self._q_cache_v_quant: Optional[torch.Tensor] = None
        self._q_cache_k_scale: Optional[torch.Tensor] = None
        self._q_cache_k_zp: Optional[torch.Tensor] = None
        self._q_cache_v_scale: Optional[torch.Tensor] = None
        self._q_cache_v_zp: Optional[torch.Tensor] = None
        self._q_cache_ids: Optional[torch.Tensor] = None
        self._q_cache_scores: Optional[torch.Tensor] = None
        self.q_retired_meta: Dict = {}
        self.rotary_emb: Optional[torch.nn.Module] = None

    def to(self, device: torch.device) -> QuantizationManager:
        if self._q_cache_k_quant is not None: self._q_cache_k_quant = self._q_cache_k_quant.to(device)
        if self._q_cache_v_quant is not None: self._q_cache_v_quant = self._q_cache_v_quant.to(device)
        if self._q_cache_k_scale is not None: self._q_cache_k_scale = self._q_cache_k_scale.to(device)
        if self._q_cache_k_zp is not None: self._q_cache_k_zp = self._q_cache_k_zp.to(device)
        if self._q_cache_v_scale is not None: self._q_cache_v_scale = self._q_cache_v_scale.to(device)
        if self._q_cache_v_zp is not None: self._q_cache_v_zp = self._q_cache_v_zp.to(device)
        if self._q_cache_ids is not None: self._q_cache_ids = self._q_cache_ids.to(device)
        if self._q_cache_scores is not None: self._q_cache_scores = self._q_cache_scores.to(device)
        return self

    def set_rotary_emb(self, rotary_emb: torch.nn.Module) -> None:
        self.rotary_emb = rotary_emb

    # --- Properties ---
    @property
    def q_cache_k_quant(self): return self._q_cache_k_quant
    @property
    def q_cache_k_scale(self): return self._q_cache_k_scale
    @property
    def q_cache_k_zp(self):    return self._q_cache_k_zp
    @property
    def q_cache_v_quant(self): return self._q_cache_v_quant
    @property
    def q_cache_v_scale(self): return self._q_cache_v_scale
    @property
    def q_cache_v_zp(self):    return self._q_cache_v_zp
    @property
    def q_cache_ids(self):     return self._q_cache_ids
    @property
    def q_cache_scores(self):  return self._q_cache_scores

    # --- store_windows ---
    def store_windows(self, k_data, v_data, window_ids, window_scores, q_phys) -> None:
        """Quantise and store loser windows from prefill."""
        self._q_cache_k_quant, self._q_cache_k_scale, self._q_cache_k_zp = \
            quantize_k_per_window(k_data, self.quant_bit_width)
        self._q_cache_v_quant, self._q_cache_v_scale, self._q_cache_v_zp = \
            quantize_v_per_window(v_data, self.quant_bit_width)
        self._q_cache_ids = window_ids.float()
        self._q_cache_scores = window_scores

    # --- accumulate_scores ---
    def accumulate_scores(self, q_attn_scores, omega: int) -> None:
        if q_attn_scores is None or self._q_cache_scores is None or self._q_cache_ids is None:
            return
        q_per_token = q_attn_scores[0, :, 0, :]
        q_tokens_total = self._q_cache_ids.shape[1] * omega
        if q_per_token.shape[1] >= q_tokens_total:
            q_per_token = q_per_token[:, :q_tokens_total]
            q_per_window = q_per_token.view(
                self.num_heads, self._q_cache_ids.shape[1], omega
            ).sum(dim=2)
            self._q_cache_scores = self._q_cache_scores + q_per_window.to(self._q_cache_scores.dtype)

    # --- get_promoted_windows ---
    def get_promoted_windows(self, winning_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        promoted_k: Dict = {h: [] for h in range(self.num_heads)}
        promoted_v: Dict = {h: [] for h in range(self.num_heads)}
        if self._q_cache_ids is None:
            return promoted_k, promoted_v

        promo_mask = (
            self._q_cache_ids.long().unsqueeze(2) == winning_ids.long().unsqueeze(1)
        ).any(dim=2)

        if promo_mask.any():
            all_k_deq = dequantize_from_quant(
                self._q_cache_k_quant, self._q_cache_k_scale,
                self._q_cache_k_zp, self.quant_bit_width
            )
            all_v_deq = dequantize_from_quant(
                self._q_cache_v_quant, self._q_cache_v_scale,
                self._q_cache_v_zp, self.quant_bit_width
            )
            promo_heads, promo_qis = promo_mask.nonzero(as_tuple=True)
            promo_wids = self._q_cache_ids[promo_heads, promo_qis].long().tolist()
            for ph, pqi, pwid in zip(promo_heads.tolist(), promo_qis.tolist(), promo_wids):
                promoted_k[ph].append((pwid, all_k_deq[ph, pqi]))
                promoted_v[ph].append((pwid, all_v_deq[ph, pqi]))
                self.q_retired_meta[(pwid, ph)] = {
                    "k_scale": self._q_cache_k_scale[ph, pqi].detach().clone(),
                    "k_zp":    self._q_cache_k_zp[ph, pqi].detach().clone(),
                    "v_scale": self._q_cache_v_scale[ph, pqi].detach().clone(),
                    "v_zp":    self._q_cache_v_zp[ph, pqi].detach().clone(),
                }
            del all_k_deq, all_v_deq
        return promoted_k, promoted_v

    # --- rebuild ---
    def rebuild(self, new_loser_ids, new_loser_scores, past_key_values,
                pre_block_wids, seq_len, omega, layer_idx) -> None:
        if new_loser_ids is None:
            # No new losers — clear Q-cache
            self._clear_cache()
            return

        device = new_loser_ids.device
        new_q_count = new_loser_ids.shape[1]
        head_dim = past_key_values[0].shape[-1]
        dtype_fp = past_key_values[0].dtype
        qbl = self._quant_bytes_len

        new_k_quant  = torch.zeros(self.num_heads, new_q_count, omega, qbl,      device=device, dtype=torch.uint8)
        new_v_quant  = torch.zeros(self.num_heads, new_q_count, omega, qbl,      device=device, dtype=torch.uint8)
        new_k_scale  = torch.zeros(self.num_heads, new_q_count, 1,    head_dim,  device=device, dtype=dtype_fp)
        new_k_zp     = torch.zeros(self.num_heads, new_q_count, 1,    head_dim,  device=device, dtype=dtype_fp)
        new_v_scale  = torch.zeros(self.num_heads, new_q_count, omega, 1,        device=device, dtype=dtype_fp)
        new_v_zp     = torch.zeros(self.num_heads, new_q_count, omega, 1,        device=device, dtype=dtype_fp)

        # --- Path A: Retained from old Q-cache ---
        if self._q_cache_ids is not None:
            retained_match = (new_loser_ids.unsqueeze(2) == self._q_cache_ids.unsqueeze(1))
            retained_any   = retained_match.any(dim=2)
            retained_old_idx = retained_match.to(torch.uint8).argmax(dim=2)
            h_idx = torch.arange(self.num_heads, device=device).unsqueeze(1).expand_as(retained_old_idx)
            if retained_any.any():
                new_k_quant[retained_any] = self._q_cache_k_quant[h_idx[retained_any], retained_old_idx[retained_any]]
                new_v_quant[retained_any] = self._q_cache_v_quant[h_idx[retained_any], retained_old_idx[retained_any]]
                new_k_scale[retained_any] = self._q_cache_k_scale[h_idx[retained_any], retained_old_idx[retained_any]]
                new_k_zp[retained_any]    = self._q_cache_k_zp[h_idx[retained_any], retained_old_idx[retained_any]]
                new_v_scale[retained_any] = self._q_cache_v_scale[h_idx[retained_any], retained_old_idx[retained_any]]
                new_v_zp[retained_any]    = self._q_cache_v_zp[h_idx[retained_any], retained_old_idx[retained_any]]
        else:
            retained_any = torch.zeros(self.num_heads, new_q_count, device=device, dtype=torch.bool)

        # --- Paths B+C: Non-retained ---
        not_retained = ~retained_any
        if not_retained.any():
            nr_h, nr_qi = not_retained.nonzero(as_tuple=True)
            nr_wids = new_loser_ids[nr_h, nr_qi].long()
            nr_count = nr_h.shape[0]
            num_old_blocks = pre_block_wids.shape[1] if pre_block_wids.numel() > 0 else 0

            if num_old_blocks > 0 and nr_count > 0:
                nr_block_wids = pre_block_wids[nr_h]
                nr_match  = (nr_block_wids == nr_wids.unsqueeze(1))
                nr_found  = nr_match.any(dim=1)
                nr_slot   = nr_match.to(torch.uint8).argmax(dim=1)
                # Physical start = sink_tokens + block_slot * omega
                # pre_block_wids was built from logical_id_map[:, sink_tokens + slot*omega]
                # so nr_slot is the block index in the compressed cache, post-sink region
                nr_phys_start = self.sink_tokens + nr_slot * omega
            else:
                nr_found = torch.zeros(nr_count, device=device, dtype=torch.bool)
                nr_phys_start = torch.zeros(nr_count, device=device, dtype=torch.long)

            nr_h_list    = nr_h.tolist()
            nr_qi_list   = nr_qi.tolist()
            nr_wids_list = nr_wids.tolist()
            nr_found_list = nr_found.tolist() if nr_count > 0 else []

            # Split into Path B (fresh) and Path C (archived meta)
            path_b, path_c = [], []
            for idx in range(nr_count):
                if (nr_wids_list[idx], nr_h_list[idx]) in self.q_retired_meta:
                    path_c.append(idx)
                else:
                    path_b.append(idx)

            offsets_om = torch.arange(omega, device=device, dtype=torch.long)

            # Path B: batch gather + quantise
            if path_b:
                pb_idx = torch.tensor(path_b, device=device, dtype=torch.long)
                pb_h   = nr_h[pb_idx];  pb_qi = nr_qi[pb_idx]
                pb_found = nr_found[pb_idx]
                pb_phys  = nr_phys_start[pb_idx]
                pb_positions = pb_phys.unsqueeze(1) + offsets_om.unsqueeze(0)
                pb_positions = pb_positions.clamp(0, seq_len - 1)
                pb_gather_idx = pb_positions.unsqueeze(-1).expand(-1, -1, head_dim)
                pb_k_heads = past_key_values[0][0, pb_h]
                pb_v_heads = past_key_values[1][0, pb_h]
                pb_k_data  = torch.gather(pb_k_heads, 1, pb_gather_idx)
                pb_v_data  = torch.gather(pb_v_heads, 1, pb_gather_idx)
                if not pb_found.all():
                    pb_k_data[~pb_found] = 0
                    pb_v_data[~pb_found] = 0

                pb_kq, pb_ks, pb_kz = quantize_k_per_window(pb_k_data.unsqueeze(1), self.quant_bit_width)
                pb_vq, pb_vs, pb_vz = quantize_v_per_window(pb_v_data.unsqueeze(1), self.quant_bit_width)
                new_k_quant[pb_h, pb_qi] = pb_kq[:, 0]
                new_v_quant[pb_h, pb_qi] = pb_vq[:, 0]
                new_k_scale[pb_h, pb_qi] = pb_ks[:, 0]
                new_k_zp[pb_h, pb_qi]    = pb_kz[:, 0]
                new_v_scale[pb_h, pb_qi] = pb_vs[:, 0]
                new_v_zp[pb_h, pb_qi]    = pb_vz[:, 0]
                del pb_k_heads, pb_v_heads, pb_k_data, pb_v_data

            # Path C: archived meta — re-quantize using stored scale/zp.
            # Keys are stored with their original absolute RoPE positions
            # (no un-rotation needed since we no longer use the rerotation scheme).
            for idx in path_c:
                h_val  = nr_h_list[idx];  qi_val = nr_qi_list[idx]
                wid_val = nr_wids_list[idx]
                if nr_found_list[idx]:
                    ps  = int(nr_phys_start[idx].item())
                    k_fp = past_key_values[0][0, h_val, ps:ps + omega]
                    v_fp = past_key_values[1][0, h_val, ps:ps + omega]
                else:
                    k_fp = torch.zeros(omega, head_dim, device=device, dtype=dtype_fp)
                    v_fp = torch.zeros(omega, head_dim, device=device, dtype=dtype_fp)
                meta = self.q_retired_meta[(wid_val, h_val)]
                ks = meta["k_scale"].to(device); kz = meta["k_zp"].to(device)
                vs = meta["v_scale"].to(device); vz = meta["v_zp"].to(device)
                if self.quant_bit_width == 8:
                    k_q = torch.round((k_fp.unsqueeze(0) - kz) / ks).clamp(0, 255).to(torch.uint8)
                    v_q = torch.round((v_fp - vz) / vs).clamp(0, 255).to(torch.uint8)
                    new_k_quant[h_val, qi_val] = k_q.squeeze(0)
                    new_v_quant[h_val, qi_val] = v_q
                else:
                    k_q = torch.round((k_fp.unsqueeze(0) - kz) / ks).clamp(0, 15).to(torch.uint8)
                    v_q = torch.round((v_fp - vz) / vs).clamp(0, 15).to(torch.uint8)
                    new_k_quant[h_val, qi_val] = ((k_q[..., 0::2] << 4) | k_q[..., 1::2]).squeeze(0)
                    new_v_quant[h_val, qi_val] = (v_q[..., 0::2] << 4) | v_q[..., 1::2]
                new_k_scale[h_val, qi_val, 0] = ks.squeeze(0)
                new_k_zp[h_val, qi_val, 0]    = kz.squeeze(0)
                new_v_scale[h_val, qi_val]     = vs.view(omega, 1)
                new_v_zp[h_val, qi_val]        = vz.view(omega, 1)

        self._q_cache_k_quant  = new_k_quant
        self._q_cache_v_quant  = new_v_quant
        self._q_cache_k_scale  = new_k_scale
        self._q_cache_k_zp     = new_k_zp
        self._q_cache_v_scale  = new_v_scale
        self._q_cache_v_zp     = new_v_zp
        self._q_cache_ids      = new_loser_ids.float()
        self._q_cache_scores   = new_loser_scores

    def _clear_cache(self):
        self._q_cache_k_quant = self._q_cache_v_quant = None
        self._q_cache_k_scale = self._q_cache_k_zp    = None
        self._q_cache_v_scale = self._q_cache_v_zp    = None
        self._q_cache_ids     = self._q_cache_scores  = None

    def reset(self) -> None:
        self._clear_cache()
        self.q_retired_meta = {}
        self._quant_bytes_len = self.head_dim if self.quant_bit_width == 8 else (self.head_dim // 2)


# ---------------------------------------------------------------------------
# No-op implementation
# ---------------------------------------------------------------------------

class NoOpQuantizationManager(BaseQuantizationManager):
    """Quantization manager used when Q_RATIO == 0.

    All properties return None and all methods are safe no-ops, eliminating
    every ``if self.q_windows_count > 0:`` branch from the coordinator.
    """

    @property
    def q_cache_k_quant(self): return None
    @property
    def q_cache_k_scale(self): return None
    @property
    def q_cache_k_zp(self):    return None
    @property
    def q_cache_v_quant(self): return None
    @property
    def q_cache_v_scale(self): return None
    @property
    def q_cache_v_zp(self):    return None
    @property
    def q_cache_ids(self):     return None
    @property
    def q_cache_scores(self):  return None

    def set_rotary_emb(self, rotary_emb: torch.nn.Module) -> None:
        pass

    def to(self, device: torch.device) -> NoOpQuantizationManager:
        return self

    def store_windows(self, k_data, v_data, window_ids, window_scores, q_phys) -> None:
        pass

    def get_promoted_windows(self, winning_ids):
        return {}, {}

    def accumulate_scores(self, q_attn_scores, omega: int) -> None:
        pass

    def rebuild(self, new_loser_ids, new_loser_scores, past_key_values,
                pre_block_wids, seq_len, omega, layer_idx) -> None:
        pass

    def reset(self) -> None:
        pass
