"""
kv_cache/eviction_manager.py
-----------------------------
EvictionManager — owns all eviction-related state and window-selection logic
for one STICKYKVCache_LayerWise layer.

State managed here:
  window_scores          [H, max_windows, 3] float32
  running_attention_votes [H, max_context]   float32
  local_history          [H, max_windows]    float32
  logical_id_map         [H, compressed_len] long   (None before prefill)

Counters:
  tokens_since_last_review, num_of_tokens_without_eviction, gen_step,
  prompt_boundary, _dynamic_local_count
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .eviction import evict_from_window_scores


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PrefillEvictionResult:
    loser_ids: Optional[torch.Tensor]    # [H, q_count] or None
    loser_scores: Optional[torch.Tensor] # [H, q_count] or None


@dataclass
class DecodeEvictionResult:
    winner_ids: torch.Tensor             # [H, curr_k] float32
    winner_scores: torch.Tensor          # [H, curr_k] float32
    curr_k: int
    loser_ids: Optional[torch.Tensor]    # [H, q_count] or None
    loser_scores: Optional[torch.Tensor]
    scoreboard: torch.Tensor             # [H, max_windows] — for local_history update
    pre_compressed_len: int
    pre_num_old_blocks: int
    pre_block_wids: torch.Tensor         # [H, num_old_blocks] long
    local_id_start: int
    local_windows: int


# ---------------------------------------------------------------------------
# EvictionManager
# ---------------------------------------------------------------------------

class EvictionManager:
    """Manages window-score tracking and eviction decisions for one cache layer."""

    def __init__(
        self,
        num_heads: int,
        max_windows: int,
        max_context: int,
        omega: int,
        sink_tokens: int,
        window_to_token_map: torch.Tensor,  # [max_windows, omega] long
        sink_indices: torch.Tensor,          # [sink_tokens] long
        device: torch.device,
    ):
        self.num_heads = num_heads
        self.omega = omega
        self.sink_tokens = sink_tokens

        # Shared read-only index buffers (owned by coordinator, referenced here)
        self.window_to_token_map = window_to_token_map
        self.sink_indices = sink_indices

        # Score state
        self.window_scores = torch.full(
            (num_heads, max_windows, 3), float("nan"), dtype=torch.float32, device=device
        )
        self.running_attention_votes = torch.zeros(
            (num_heads, max_context), dtype=torch.float32, device=device
        )
        self.local_history = torch.zeros(
            (num_heads, max_windows), dtype=torch.float32, device=device
        )
        self._head_indices = torch.arange(num_heads, device=device)

        # Mapping state
        self.logical_id_map: Optional[torch.Tensor] = None

        # Counters
        self.tokens_since_last_review: int = 0
        self.num_of_tokens_without_eviction: int = 0
        self.gen_step: int = 0
        self.prompt_boundary = [-1] * num_heads
        self._dynamic_local_count: int = 0

    def to(self, device: torch.device) -> EvictionManager:
        """Migrate all internal state tensors to the specified device."""
        self.window_scores = self.window_scores.to(device)
        self.running_attention_votes = self.running_attention_votes.to(device)
        self.local_history = self.local_history.to(device)
        self._head_indices = self._head_indices.to(device)
        if self.logical_id_map is not None:
            self.logical_id_map = self.logical_id_map.to(device)
        return self

    # ------------------------------------------------------------------
    # Prefill helpers
    # ------------------------------------------------------------------

    def tally_prefill_scores(
        self,
        attn_score_cache: torch.Tensor,
        seq_len: int,
        local_num: int,
    ) -> int:
        """Populate window_scores and local_history from prefill attention.

        Returns score_end — the physical token boundary passed to
        create_mask_and_evict_from_kv_cache_prompt_stage.
        """
        sink_tokens = self.sink_tokens
        omega = self.omega
        num_heads = self.num_heads
        device = self.window_scores.device

        score_end = max(sink_tokens, seq_len - local_num)
        num_windows = max(0, (score_end - sink_tokens) // omega)
        score_end = min(sink_tokens + num_windows * omega, attn_score_cache.shape[3])
        num_windows = (score_end - sink_tokens) // omega
        score_end = sink_tokens + num_windows * omega
        self._dynamic_local_count = local_num

        if num_windows > 0:
            scores_slice = attn_score_cache[0, :, :seq_len, sink_tokens:score_end]
            obs_sum = scores_slice.sum(dim=1)
            win_scores = (
                obs_sum.view(num_heads, num_windows, omega).sum(dim=2).to(torch.float32)
            )
            idx = (
                torch.arange(num_windows, device=device)
                .unsqueeze(0)
                .expand(num_heads, -1)
            )
            self.window_scores[self._head_indices.unsqueeze(1), idx, 0] = win_scores
            self.window_scores[self._head_indices.unsqueeze(1), idx, 1] = idx.float()
            self.window_scores[self._head_indices.unsqueeze(1), idx, 2] = idx.float()

        # Seed local_history from the full prompt attention
        total_prompt_windows = max(0, (seq_len - sink_tokens) // omega)
        if total_prompt_windows > 0:
            full_review_end = sink_tokens + total_prompt_windows * omega
            actual_full_review = min(full_review_end, attn_score_cache.shape[3])
            total_prompt_windows = (actual_full_review - sink_tokens) // omega
            actual_full_review = sink_tokens + total_prompt_windows * omega
            if total_prompt_windows > 0:
                full_slice = attn_score_cache[0, :, :seq_len, sink_tokens:actual_full_review]
                full_obs_sum = full_slice.sum(dim=1)
                full_win_scores = (
                    full_obs_sum.view(num_heads, total_prompt_windows, omega)
                    .sum(dim=2)
                    .to(torch.float32)
                )
                idx_full = torch.arange(
                    total_prompt_windows, device=device, dtype=torch.long
                )
                self.local_history[:, idx_full] = full_win_scores

        return score_end

    def get_prefill_eviction(
        self, k_windows: int, q_windows_count: int
    ) -> PrefillEvictionResult:
        """Select top-k windows at prefill; returns loser windows for Q-cache."""
        q_loser_ids, q_loser_scores = evict_from_window_scores(
            self.window_scores, k_windows, q_windows_count
        )
        return PrefillEvictionResult(loser_ids=q_loser_ids, loser_scores=q_loser_scores)

    def commit_prefill_result(self, survivor_ids: torch.Tensor) -> None:
        """Build logical_id_map from physical survivor indices."""
        sink_tokens = self.sink_tokens
        omega = self.omega
        self.logical_id_map = torch.where(
            survivor_ids >= sink_tokens,
            (survivor_ids - sink_tokens) // omega,
            torch.full_like(survivor_ids, -1),
        ).to(torch.long)

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def tally_decode_scores(
        self, attn_score_cache: torch.Tensor, seq_len: int
    ) -> None:
        """Accumulate one generation step's attention votes."""
        self.running_attention_votes[:, :seq_len] += attn_score_cache[0, :, 0, :seq_len]
        self.tokens_since_last_review += 1

    def run_decode_cycle(
        self,
        k_windows: int,
        q_windows_count: int,
        local_tokens_count: int,
        q_cache_ids: Optional[torch.Tensor],
        q_cache_scores: Optional[torch.Tensor],
        seq_len: int,
    ) -> DecodeEvictionResult:
        """Full decode eviction cycle — called every omega tokens.

        Includes the per-head ``already_tracked_per_head`` dedup guard
        (fixes the NameError bug present in the original cumulative cache.py).
        """
        device = self.window_scores.device
        num_heads = self.num_heads
        omega = self.omega
        sink_tokens = self.sink_tokens

        compressed_len = self.logical_id_map.shape[1]
        compressed_votes = self.running_attention_votes[:, :compressed_len]
        logical_ids = self.logical_id_map

        # --- Build scoreboard ---
        is_chunk_token = logical_ids >= 0
        routed_votes = torch.where(
            is_chunk_token, compressed_votes, torch.zeros_like(compressed_votes)
        )
        safe_logical_ids = torch.where(
            is_chunk_token, logical_ids, torch.zeros_like(logical_ids)
        ).long()
        scoreboard = torch.zeros(
            (num_heads, self.window_scores.shape[1]), device=device, dtype=torch.float32
        )
        scoreboard.scatter_add_(1, safe_logical_ids, routed_votes)

        # Route votes for the new omega tokens not yet in logical_id_map
        if seq_len > compressed_len:
            n_new = seq_len - compressed_len
            new_tok_votes = self.running_attention_votes[:, compressed_len:seq_len]
            js = torch.arange(n_new, device=device, dtype=torch.long)
            raw_new_lids = (
                (self.num_of_tokens_without_eviction - omega + js - sink_tokens) // omega
            )
            valid_new = (raw_new_lids >= 0) & (raw_new_lids < scoreboard.shape[1])
            if valid_new.any():
                scoreboard.scatter_add_(
                    1,
                    raw_new_lids[valid_new].unsqueeze(0).expand(num_heads, -1),
                    new_tok_votes[:, valid_new],
                )

        # --- Read existing window scores ---
        valid_mask = ~torch.isnan(self.window_scores[:, :, 1])
        # WARN-2 fix: use per-head max rather than cross-head min to avoid silently
        # dropping valid windows for heads that have more entries than the weakest head.
        valid_old_windows = min(k_windows, int(valid_mask.sum(dim=1).max().item()))
        raw_ids = self.window_scores[:, :valid_old_windows, 1]
        raw_scores = self.window_scores[:, :valid_old_windows, 0]
        is_valid_slot = ~torch.isnan(raw_ids)
        old_ids = raw_ids.nan_to_num(nan=0.0)
        safe_ids = old_ids.long()
        old_scores_hist = torch.nan_to_num(raw_scores, nan=0.0)
        if valid_old_windows > 0:
            old_w_gen_scores = scoreboard.gather(1, safe_ids)
            old_w_gen_scores.masked_fill_(~is_valid_slot, 0.0)
        else:
            old_w_gen_scores = old_scores_hist.new_zeros(old_scores_hist.shape)
        old_scores = old_scores_hist + old_w_gen_scores

        # --- Score the challenger (window leaving the local bubble) ---
        raw_last_id_val = (
            (self.num_of_tokens_without_eviction - sink_tokens - local_tokens_count)
            // omega
            - 1
        )
        raw_last_id_val = min(raw_last_id_val, self.window_scores.shape[1] - 1)
        has_challenger = raw_last_id_val >= 0
        last_id_val = raw_last_id_val

        if has_challenger:
            last_id_tensor = torch.full(
                (num_heads, 1), float(last_id_val), device=device, dtype=torch.float32
            )
            # BUG FIX: define already_tracked_per_head (missing in original cumulative cache.py)
            already_tracked_per_head = (old_ids.long() == last_id_val).any(dim=1)

            if last_id_val < scoreboard.shape[1]:
                new_w_gen_scores = scoreboard[:, int(last_id_val)]
            else:
                new_w_gen_scores = torch.zeros(num_heads, dtype=torch.float32, device=device)

            if last_id_val < self.local_history.shape[1]:
                last_id_hist_scores = self.local_history[:, last_id_val].clone()
                self.local_history[:, last_id_val] = torch.where(
                    already_tracked_per_head,
                    self.local_history[:, last_id_val],
                    torch.zeros_like(self.local_history[:, last_id_val]),
                )
            else:
                last_id_hist_scores = torch.zeros(num_heads, dtype=torch.float32, device=device)

            new_w_total_scores = new_w_gen_scores + last_id_hist_scores
            new_w_total_scores = torch.where(
                already_tracked_per_head,
                torch.full_like(new_w_total_scores, float("-inf")),
                new_w_total_scores,
            )
            competing_ids = torch.cat([old_ids, last_id_tensor], dim=1)
            competing_scores = torch.cat(
                [old_scores, new_w_total_scores.unsqueeze(1)], dim=1
            )
        else:
            competing_ids = old_ids
            competing_scores = old_scores

        # Include q-cache windows in competition
        if q_windows_count > 0 and q_cache_ids is not None:
            competing_ids = torch.cat([competing_ids, q_cache_ids], dim=1)
            competing_scores = torch.cat([competing_scores, q_cache_scores], dim=1)

        # --- Top-k: winners (BF16 sticky) ---
        curr_k = min(k_windows, competing_scores.shape[1])
        top_v, top_i = torch.topk(competing_scores, curr_k, dim=1, largest=True)
        surviving_ids = torch.gather(competing_ids, 1, top_i)
        sort_idx = torch.argsort(surviving_ids, dim=1)
        final_v = torch.gather(top_v, 1, sort_idx)
        final_ids = torch.gather(surviving_ids, 1, sort_idx)

        # --- Top-q: losers (new Q-cache) ---
        new_q_loser_ids = None
        new_q_loser_scores = None
        if q_windows_count > 0:
            loser_scores = competing_scores.scatter(1, top_i, float("-inf"))
            num_remaining = int((loser_scores > float("-inf")).sum(dim=1).min().item())
            if num_remaining > 0:
                q_count = min(q_windows_count, num_remaining)
                q_top_v, q_top_i = torch.topk(loser_scores, q_count, dim=1, largest=True)
                new_q_loser_ids = torch.gather(competing_ids, 1, q_top_i)
                new_q_loser_scores = q_top_v

        # --- Local history start / windows count ---
        local_windows = (local_tokens_count + omega - 1) // omega
        if has_challenger:
            local_id_start = last_id_val + 1
        else:
            local_id_start = max(
                0,
                (self.num_of_tokens_without_eviction - sink_tokens - local_tokens_count)
                // omega,
            )

        # --- Precompute block-boundary data (reused in Q-cache rebuild + physical eviction) ---
        pre_compressed_len = compressed_len
        pre_num_old_blocks = max(
            0, (pre_compressed_len - sink_tokens - local_tokens_count) // omega
        )
        if pre_num_old_blocks > 0:
            pre_block_starts = sink_tokens + torch.arange(
                pre_num_old_blocks, device=device, dtype=torch.long
            ) * omega
            pre_block_wids = self.logical_id_map[:, pre_block_starts]
        else:
            pre_block_wids = torch.zeros(num_heads, 0, device=device, dtype=torch.long)

        return DecodeEvictionResult(
            winner_ids=final_ids,
            winner_scores=final_v,
            curr_k=curr_k,
            loser_ids=new_q_loser_ids,
            loser_scores=new_q_loser_scores,
            scoreboard=scoreboard,
            pre_compressed_len=pre_compressed_len,
            pre_num_old_blocks=pre_num_old_blocks,
            pre_block_wids=pre_block_wids,
            local_id_start=local_id_start,
            local_windows=local_windows,
        )

    def commit_decode_result(
        self,
        result: DecodeEvictionResult,
        new_logical_id_map: torch.Tensor,
    ) -> None:
        """Update window_scores and logical_id_map after physical cache rebuild."""
        curr_k = result.curr_k
        self.window_scores.fill_(float("nan"))
        self.window_scores[:, :curr_k, 0] = result.winner_scores
        self.window_scores[:, :curr_k, 1] = result.winner_ids
        self.window_scores[:, :curr_k, 2] = result.winner_ids
        self.logical_id_map = new_logical_id_map

    def update_local_history(self, result: DecodeEvictionResult) -> None:
        """Accumulate local-zone votes into local_history."""
        if result.local_windows <= 0:
            return
        device = self.local_history.device
        ids = torch.arange(
            result.local_id_start,
            result.local_id_start + result.local_windows,
            device=device,
            dtype=torch.long,
        )
        valid = (ids >= 0) & (ids < self.local_history.shape[1])
        if valid.any():
            ids_valid = ids[valid]
            self.local_history[:, ids_valid] += result.scoreboard[:, ids_valid]

    def reset_decode_counters(self, seq_len: int) -> None:
        """Zero running_attention_votes and reset review counter after a cycle."""
        self.running_attention_votes[:, :seq_len].zero_()
        self.tokens_since_last_review = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all eviction state between documents / sequences."""
        self.gen_step = 0
        self.num_of_tokens_without_eviction = 0
        self.tokens_since_last_review = 0
        self._dynamic_local_count = 0
        self.logical_id_map = None
        self.prompt_boundary = [-1] * self.num_heads
        self.running_attention_votes.zero_()
        self.window_scores.fill_(float("nan"))
        self.local_history.zero_()
