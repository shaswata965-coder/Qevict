"""
models/sticky_cache.py
----------------------
StickyCache — a minimal Cache subclass that carries our evicted KV tuples
through Hugging Face's generate() loop without triggering DynamicCache's
append/grow logic.

Design:
  - Each layer slot holds a (keys, values) tuple or None.
  - get_kv() returns None if the slot is empty (prefill step).
  - set_kv() stores the post-eviction tuple for the next decode step.
  - All HF Cache abstract methods are implemented as no-ops or stubs to
    satisfy transformers ≥ 4.46 without breaking our eviction lifecycle.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch

try:
    from transformers import Cache
except ImportError:
    # Fallback for environments where transformers is unavailable at import time
    Cache = object  # type: ignore


class StickyCache(Cache):
    """Transparent KV-tuple carrier for the Sticky eviction pipeline.

    Unlike DynamicCache, this class does NOT concatenate tensors on update().
    It simply holds whatever (k, v) tuple our attention module writes after
    each eviction cycle.
    """

    def __init__(self, num_layers: int = 0):
        super().__init__()
        # Per-layer slots: None = empty (prefill step)
        self._kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers
        self._num_layers = num_layers

    # ------------------------------------------------------------------
    # Our API — used by module.py / module_flash.py
    # ------------------------------------------------------------------

    def get_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the stored (k, v) tuple for this layer, or None at prefill."""
        if layer_idx < len(self._kv):
            return self._kv[layer_idx]
        return None

    def set_kv(self, layer_idx: int, kv: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Store post-eviction (k, v) tuple for this layer."""
        while len(self._kv) <= layer_idx:
            self._kv.append(None)
        self._kv[layer_idx] = kv

    # ------------------------------------------------------------------
    # HF Cache abstract method implementations (≥ 4.36)
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """HF calls this during forward().  We ignore it — our attention
        module manages the cache directly via get_kv / set_kv."""
        existing = self.get_kv(layer_idx)
        if existing is not None:
            return existing
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return physical sequence length for the given layer (or 0 if empty)."""
        kv = self.get_kv(layer_idx)
        if kv is not None:
            return kv[0].shape[-2]
        return 0

    def get_max_length(self) -> Optional[int]:
        """Unbounded — we don't enforce a static max length."""
        return None

    def __len__(self) -> int:
        return self._num_layers

    # ------------------------------------------------------------------
    # DynamicCache-compatible property stubs
    # HF's generate() may inspect key_cache / value_cache on the returned
    # cache object.  Provide read-only list views so those accesses don't crash.
    # ------------------------------------------------------------------

    @property
    def key_cache(self) -> List[torch.Tensor]:
        return [kv[0] if kv is not None else torch.empty(0) for kv in self._kv]

    @property
    def value_cache(self) -> List[torch.Tensor]:
        return [kv[1] if kv is not None else torch.empty(0) for kv in self._kv]
