"""
kv_cache/base_quantization_manager.py
--------------------------------------
Abstract base class for the quantized eviction side-cache.

Concrete implementations:
  QuantizationManager     — active path (Q_RATIO > 0)
  NoOpQuantizationManager — inactive path (Q_RATIO == 0)

Both live in quantize_manager.py.  This file is kept separate so third-party
strategies (e.g. FP8, INT3) can subclass BaseQuantizationManager without
importing anything from the concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch


class BaseQuantizationManager(ABC):
    """Strategy interface for the quantized eviction side-cache.

    Every method has a clearly typed signature so alternative strategies
    (INT4, FP8, …) can be plugged in by subclassing and implementing all
    abstract members.
    """

    # ------------------------------------------------------------------
    # Read-only state properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def q_cache_k_quant(self) -> Optional[torch.Tensor]:
        """INT-quantised K data, shape [H, W, omega, quant_bytes] uint8."""

    @property
    @abstractmethod
    def q_cache_k_scale(self) -> Optional[torch.Tensor]:
        """K dequantisation scale, shape [H, W, 1, D]."""

    @property
    @abstractmethod
    def q_cache_k_zp(self) -> Optional[torch.Tensor]:
        """K dequantisation zero-point, shape [H, W, 1, D]."""

    @property
    @abstractmethod
    def q_cache_v_quant(self) -> Optional[torch.Tensor]:
        """INT-quantised V data, shape [H, W, omega, quant_bytes] uint8."""

    @property
    @abstractmethod
    def q_cache_v_scale(self) -> Optional[torch.Tensor]:
        """V dequantisation scale, shape [H, W, omega, 1]."""

    @property
    @abstractmethod
    def q_cache_v_zp(self) -> Optional[torch.Tensor]:
        """V dequantisation zero-point, shape [H, W, omega, 1]."""

    @property
    @abstractmethod
    def q_cache_ids(self) -> Optional[torch.Tensor]:
        """Logical window IDs currently stored in the Q-cache, [H, W] float32."""

    @property
    @abstractmethod
    def q_cache_scores(self) -> Optional[torch.Tensor]:
        """Cumulative attention scores for Q-cache windows, [H, W] float32."""

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    @abstractmethod
    def set_rotary_emb(self, rotary_emb: torch.nn.Module) -> None:
        """Set the RoPE module for un-rotating keys before quantization."""

    @abstractmethod
    def store_windows(
        self,
        k_data: torch.Tensor,
        v_data: torch.Tensor,
        window_ids: torch.Tensor,
        window_scores: torch.Tensor,
        q_phys: torch.Tensor,
    ) -> None:
        """Quantise and store a set of evicted KV windows.

        Called once at the end of the prefill stage.

        Parameters
        ----------
        k_data       : [H, W, omega, D] float  — already-gathered KV tensors.
        v_data       : [H, W, omega, D] float
        window_ids   : [H, W] float — logical window IDs being stored.
        window_scores: [H, W] float — eviction scores for these windows.
        q_phys       : [H, W, omega] long — physical token positions for RoPE un-rotation.
        """

    @abstractmethod
    def get_promoted_windows(
        self,
        winning_ids: torch.Tensor,
    ) -> Tuple[Dict[Tuple[int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]:
        """Return dequantised KV data for Q-cache windows promoted to BF16.

        Also records retired metadata (scale/zp) in ``q_retired_meta`` for
        any promoted window so Path-C of the Q-cache rebuild can reuse them.

        Parameters
        ----------
        winning_ids : [H, curr_k] float — BF16-cache window IDs after topk.

        Returns
        -------
        promoted_k : dict mapping (head, wid) → [omega, D] tensor
        promoted_v : dict mapping (head, wid) → [omega, D] tensor
        """

    @abstractmethod
    def accumulate_scores(
        self,
        q_attn_scores: Optional[torch.Tensor],
        omega: int,
    ) -> None:
        """Accumulate per-step Q-cache attention scores from joint softmax.

        Parameters
        ----------
        q_attn_scores : [1, H, 1, q_tokens] float, or None if Q-cache inactive.
        omega         : window size (tokens per window).
        """

    @abstractmethod
    def rebuild(
        self,
        new_loser_ids: Optional[torch.Tensor],
        new_loser_scores: Optional[torch.Tensor],
        past_key_values: Tuple[torch.Tensor, torch.Tensor],
        pre_block_wids: torch.Tensor,
        seq_len: int,
        omega: int,
        layer_idx: int,
    ) -> None:
        """Rebuild the Q-cache after one decode eviction cycle.

        Implements the three routing paths:
          A — retained from previous Q-cache (raw int copy)
          B — fresh from main BF16 cache (gather + quantise)
          C — archived meta available (re-quantise with stored scale/zp)

        Parameters
        ----------
        new_loser_ids   : [H, q_count] — new Q-cache occupant IDs, or None.
        new_loser_scores: [H, q_count] — scores for new occupants, or None.
        past_key_values : (K, V) tuple, shapes [1, H, S, D].
        pre_block_wids  : [H, num_old_blocks] — logical IDs of old blocks.
        seq_len         : current physical KV sequence length.
        omega           : window size.
        layer_idx       : for warning messages.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all Q-cache state (called between documents / sequences)."""
