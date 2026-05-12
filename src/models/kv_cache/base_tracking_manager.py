"""
kv_cache/base_tracking_manager.py
----------------------------------
Abstract base class for the research-analysis token-tracking ledger.

Concrete implementations:
  TrackingManager     — active path (tracking_flag == 1)
  NoOpTrackingManager — inactive path (tracking_flag == 0)

Both live in tracking_manager.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseTrackingManager(ABC):
    """Interface for the token provenance and importance tracking ledger.

    When ``tracking_flag == 0`` a ``NoOpTrackingManager`` is substituted so the
    coordinator ``__call__`` has zero ``if self.tracking_flag:`` branches.
    """

    # ------------------------------------------------------------------
    # State properties (externally accessed by eval scripts)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def token_ledger(self) -> torch.Tensor:
        """[max_context, 2 + 2*num_heads] float32 — global token metadata."""

    @property
    @abstractmethod
    def global_token_counter(self) -> torch.Tensor:
        """Scalar LongTensor — total tokens seen so far (arrival order)."""

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    @abstractmethod
    def increment(self, n: int = 1) -> None:
        """Advance global_token_counter by n (n=q_len at prefill, n=1 at decode)."""

    @abstractmethod
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
        """Write per-token metadata into the ledger after prefill eviction.

        Parameters
        ----------
        seq_len           : physical KV length after appending new tokens.
        q_len             : number of prefill tokens.
        layer_idx         : cache layer index.
        attn_score_cache  : [1, H, q_len, kv_len] attention weights.
        survivor_ids      : [H, N_survivors] physical indices kept.
        full_attn_scores  : full attention for research logging (or None).
        global_start      : global token counter value at start of prefill.
        num_new_tokens    : same as q_len (alias for clarity in callee).
        """

    @abstractmethod
    def record_decode(
        self,
        attn_score_cache: torch.Tensor,
        seq_len: int,
        layer_idx: int,
        global_start: int,
    ) -> None:
        """Update per-token metadata for a single decode step.

        Parameters
        ----------
        attn_score_cache : [1, H, 1, kv_len] attention weights.
        seq_len          : current physical KV sequence length.
        layer_idx        : cache layer index (for ledger column 1).
        global_start     : global_token_counter value before this step.
        """

    @abstractmethod
    def update_physical_positions(
        self,
        mapping: torch.Tensor,
        seq_len: int,
        running_attention_votes: torch.Tensor,
    ) -> None:
        """Remap physical-position columns in the ledger after cache rebuild.

        Parameters
        ----------
        mapping                  : [H, seq_len] float32 — maps old physical idx
                                   to new physical idx (-1 means evicted).
        seq_len                  : physical KV length before rebuild.
        running_attention_votes  : [H, max_context] — votes for evicted tokens
                                   are folded into the ledger attention score.
        """

    @abstractmethod
    def get_ledger_data(self) -> dict:
        """Return a structured view of the active ledger slice.

        Returns dict with keys: global_id, layer_id, physical_id,
        attention_score (matches ledger.get_ledger_data() contract).
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracking state between documents / sequences."""
