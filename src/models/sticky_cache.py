"""
models/sticky_cache.py
----------------------
StickyCache — a minimal Cache subclass that carries our evicted KV tuples
through Hugging Face's generate() loop without triggering DynamicCache's
append/grow logic.

Design:
  - Each layer slot holds a (keys, values) pair via two real mutable lists:
    key_cache and value_cache.  These are the SAME names HF uses on
    DynamicCache, so any HF code that reads cache.key_cache[layer_idx]
    will work transparently.
  - get_kv() returns None if the slot is empty (prefill step).
  - set_kv() stores the post-eviction tensors for the next decode step.
  - All HF Cache abstract methods are implemented as no-ops or stubs to
    satisfy transformers >= 4.36 without breaking our eviction lifecycle.

IMPORTANT: key_cache and value_cache MUST be real lists (not @property
that creates throwaway copies) so that ``cache.key_cache[i] = tensor``
actually persists.
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
    """Transparent KV-tensor carrier for the Sticky eviction pipeline.

    Unlike DynamicCache, this class does NOT concatenate tensors on update().
    It simply holds whatever (k, v) tensors our attention module writes after
    each eviction cycle.
    """

    def __init__(self, num_layers: int = 0):
        super().__init__()
        self._num_layers = num_layers

        # Real mutable lists — same attribute names as DynamicCache so that
        # both HF internals and our own code can read/write via
        #   cache.key_cache[layer_idx]  and  cache.value_cache[layer_idx]
        self.key_cache: List[torch.Tensor] = [torch.empty(0) for _ in range(num_layers)]
        self.value_cache: List[torch.Tensor] = [torch.empty(0) for _ in range(num_layers)]

        # _seen_tokens tracks the TRUE total sequence length (not compressed).
        # HF's generate loop reads this to compute cache_position and causal
        # masks.  We update it from the attention module at layer 0.
        self._seen_tokens = 0

    # ------------------------------------------------------------------
    # Our API — used by module.py / module_flash.py
    # ------------------------------------------------------------------

    def get_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the stored (k, v) tuple for this layer, or None at prefill."""
        if layer_idx < len(self.key_cache):
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
            if k.numel() > 0:
                return (k, v)
        return None

    def set_kv(self, layer_idx: int, kv: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Store post-eviction (k, v) tensors for this layer."""
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0))
            self.value_cache.append(torch.empty(0))
        self.key_cache[layer_idx] = kv[0]
        self.value_cache[layer_idx] = kv[1]

    # ------------------------------------------------------------------
    # HF Cache abstract method implementations (>= 4.36)
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """HF calls this during forward().  We ignore it — our attention
        module manages the cache directly via set_kv / key_cache writes."""
        existing = self.get_kv(layer_idx)
        if existing is not None:
            return existing
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return physical sequence length for the given layer (or 0 if empty)."""
        if layer_idx < len(self.key_cache):
            k = self.key_cache[layer_idx]
            if k.numel() > 0:
                return k.shape[-2]
        return 0

    def get_max_length(self) -> Optional[int]:
        """Unbounded — we don't enforce a static max length."""
        return None

    def __len__(self) -> int:
        return self._num_layers
