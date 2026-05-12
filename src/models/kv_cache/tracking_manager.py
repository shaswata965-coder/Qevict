"""
kv_cache/tracking_manager.py
-----------------------------
Concrete implementations of BaseTrackingManager:

  TrackingManager     — active path (tracking_flag == 1)
  NoOpTrackingManager — inactive path (tracking_flag == 0)

When tracking is disabled the NoOp is used, eliminating all
``if self.tracking_flag:`` branches from the coordinator __call__.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base_tracking_manager import BaseTrackingManager
from .ledger import get_ledger_data as _get_ledger_data


# ---------------------------------------------------------------------------
# Active implementation
# ---------------------------------------------------------------------------

class TrackingManager(BaseTrackingManager):
    """Maintains the token provenance and cumulative importance ledger."""

    def __init__(self, max_context: int, num_heads: int, device: torch.device):
        self.num_heads = num_heads

        self._global_token_counter = torch.tensor(0, dtype=torch.long, device=device)
        self._token_ledger = torch.full(
            (max_context, 2 + 2 * num_heads), -1.0, dtype=torch.float32, device=device
        )
        self._global_score_history = torch.full(
            (max_context, num_heads), -1.0, dtype=torch.float32, device=device
        )
        self.prefill_attention_matrix = None

    @property
    def token_ledger(self) -> torch.Tensor:
        return self._token_ledger

    @property
    def global_token_counter(self) -> torch.Tensor:
        return self._global_token_counter

    def increment(self, n: int = 1) -> None:
        self._global_token_counter += n

    def record_prefill(
        self,
        seq_len: int,
        q_len: int,
        layer_idx: int,
        attn_score_cache: torch.Tensor,
        survivor_ids: torch.Tensor,
        full_attn_scores: Optional[torch.Tensor],
        global_start: int,
        num_new_tokens: int,
    ) -> None:
        num_heads = self.num_heads
        ledger = self._token_ledger
        ledger_size = ledger.shape[0]
        device = attn_score_cache.device

        # Initial ledger fill (physical positions)
        for i in range(num_new_tokens):
            g_id = global_start + i
            if g_id < ledger_size:
                ledger[g_id, 0] = float(g_id)
                ledger[g_id, 1] = float(layer_idx)
                ledger[g_id, 2:2 + num_heads] = float(seq_len - q_len + i)

        # Pre-eviction importance scores
        importance_map = attn_score_cache[0, :, :seq_len, :].sum(dim=1)
        n_prefill = min(seq_len, ledger_size)
        active_mask = (
            (ledger[:n_prefill, 2:2 + num_heads] >= 0).any(dim=1)
            & (ledger[:n_prefill, 0] >= 0)
        )
        active_g_ids = torch.where(active_mask)[0]
        if active_g_ids.numel() > 0:
            phys_idx = ledger[active_g_ids, 2].long()
            valid = (phys_idx >= 0) & (phys_idx < importance_map.shape[1])
            v_gids = active_g_ids[valid]
            v_phys = phys_idx[valid]
            if v_gids.numel() > 0:
                scores_for_valid = importance_map[:, v_phys].T.float()
                ledger[v_gids, 2 + num_heads:2 + 2 * num_heads] = scores_for_valid
                self._global_score_history[v_gids, :] = scores_for_valid

        # Store prefill attention matrix for external analysis
        full_ref = full_attn_scores if full_attn_scores is not None else attn_score_cache
        self.prefill_attention_matrix = full_ref[0].detach().cpu()

        # Post-eviction: update physical positions to compressed locations
        full_importance = attn_score_cache[0, :, :seq_len, :].sum(dim=1)
        ledger[:, 2:2 + num_heads] = -1.0
        for head_idx in range(num_heads):
            clean_survivors = survivor_ids[head_idx].to(torch.long)
            g_ids = global_start + clean_survivors - (seq_len - num_new_tokens)
            valid = (g_ids >= 0) & (g_ids < ledger_size)
            if valid.any():
                valid_g = g_ids[valid]
                compressed_pos = torch.arange(
                    clean_survivors.shape[0], device=device, dtype=torch.float32
                )
                ledger[valid_g, 2 + head_idx] = compressed_pos[valid]
                valid_phys = clean_survivors[valid]
                in_range = valid_phys < full_importance.shape[1]
                if in_range.any():
                    ledger[valid_g[in_range], 2 + num_heads + head_idx] = \
                        full_importance[head_idx, valid_phys[in_range]].float()

    def record_decode(
        self,
        attn_score_cache: torch.Tensor,
        seq_len: int,
        layer_idx: int,
        global_start: int,
    ) -> None:
        num_heads = self.num_heads
        ledger = self._token_ledger
        ledger_size = ledger.shape[0]
        g_id = global_start
        if g_id < ledger_size:
            ledger[g_id, 0] = float(g_id)
            ledger[g_id, 1] = float(layer_idx)
            ledger[g_id, 2:2 + num_heads] = float(seq_len - 1)
            ledger[g_id, 2 + num_heads:2 + 2 * num_heads] = 0.0
            self._global_score_history[g_id, :] = 0.0

        n_active = min(int(self._global_token_counter.item()), ledger_size)
        if n_active > 0:
            live_mask = (ledger[:n_active, 2:2 + num_heads] >= 0).any(dim=1)
            live_g_ids = torch.where(live_mask)[0]
            if live_g_ids.numel() > 0:
                all_phys = ledger[live_g_ids, 2:2 + num_heads].long()
                kv_seq = attn_score_cache.size(-1)
                in_range = (all_phys >= 0) & (all_phys < kv_seq)
                safe_phys = all_phys.clamp(min=0, max=kv_seq - 1)
                gen_scores = attn_score_cache[0, :, 0, :]
                head_grid = torch.arange(num_heads, device=gen_scores.device).unsqueeze(0).expand(
                    live_g_ids.shape[0], -1
                )
                gathered = gen_scores[head_grid, safe_phys]
                gathered = gathered * in_range.to(gathered.dtype)
                ledger[live_g_ids, 2 + num_heads:2 + 2 * num_heads] += gathered.float()
                self._global_score_history[live_g_ids, :] += gathered.float()

    def update_physical_positions(
        self,
        mapping: torch.Tensor,
        seq_len: int,
        running_attention_votes: torch.Tensor,
    ) -> None:
        num_heads = self.num_heads
        ledger = self._token_ledger
        device = mapping.device
        n_active = min(int(self._global_token_counter.item()), ledger.shape[0])
        if n_active <= 0:
            return
        live_mask = (ledger[:n_active, 2:2 + num_heads] >= 0).any(dim=1)
        live_indices = torch.where(live_mask)[0]
        if live_indices.numel() == 0:
            return
        all_phys = ledger[live_indices, 2:2 + num_heads].long()
        valid_old = (all_phys >= 0) & (all_phys < seq_len)
        safe_phys = all_phys.clamp(min=0, max=seq_len - 1)
        head_grid = torch.arange(num_heads, device=device).unsqueeze(0).expand(
            live_indices.shape[0], -1
        )
        new_phys = mapping[head_grid, safe_phys]
        new_phys = torch.where(valid_old, new_phys, torch.full_like(new_phys, -1.0))
        is_evicted = (new_phys < 0) & valid_old
        votes = running_attention_votes[head_grid, safe_phys]
        votes_to_add = votes * is_evicted.to(votes.dtype)
        ledger[live_indices, 2 + num_heads:2 + 2 * num_heads] += votes_to_add.float()
        self._global_score_history[live_indices, :] += votes_to_add.float()
        ledger[live_indices, 2:2 + num_heads] = new_phys

    def get_ledger_data(self) -> dict:
        return _get_ledger_data(self._global_token_counter, self._token_ledger, self.num_heads)

    def reset(self) -> None:
        self._global_token_counter.zero_()
        self._token_ledger.fill_(-1.0)
        self._global_score_history.fill_(-1.0)
        self.prefill_attention_matrix = None


# ---------------------------------------------------------------------------
# No-op implementation
# ---------------------------------------------------------------------------

class NoOpTrackingManager(BaseTrackingManager):
    """Tracking manager used when tracking_flag == 0.

    Stubs return minimal tensors to satisfy hasattr checks while all
    record/update methods are safe no-ops.
    """

    def __init__(self, device: torch.device):
        # Minimal stubs so eval scripts that access these don't crash
        self._global_token_counter = torch.tensor(0, dtype=torch.long, device=device)
        # 1-row stub — eval scripts check shape[0] via global_token_counter, not directly
        self._token_ledger = torch.zeros(1, 2, dtype=torch.float32, device=device)
        # BUG-1 fix: coordinator exposes this via cache.global_score_history;
        # NoOp must define it so the property doesn't raise AttributeError.
        self._global_score_history = torch.zeros(1, 1, dtype=torch.float32, device=device)

    @property
    def token_ledger(self) -> torch.Tensor:
        return self._token_ledger

    @property
    def global_token_counter(self) -> torch.Tensor:
        return self._global_token_counter

    def increment(self, n: int = 1) -> None:
        self._global_token_counter += n

    def record_prefill(self, seq_len, q_len, layer_idx, attn_score_cache,
                       survivor_ids, full_attn_scores, global_start, num_new_tokens) -> None:
        pass

    def record_decode(self, attn_score_cache, seq_len, layer_idx, global_start) -> None:
        pass

    def update_physical_positions(self, mapping, seq_len, running_attention_votes) -> None:
        pass

    def get_ledger_data(self) -> dict:
        return {"global_id": torch.zeros(0), "layer_id": torch.zeros(0),
                "physical_id": torch.zeros(0), "attention_score": torch.zeros(0)}

    def reset(self) -> None:
        self._global_token_counter.zero_()
