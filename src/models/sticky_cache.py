"""
models/sticky_cache.py
----------------------
Cache layer and cache wrapper for the Sticky KV-cache eviction pipeline.

StickyDynamicLayer — a DynamicLayer subclass that tracks cumulative sequence
length through eviction/compression cycles.  Modelled after HF's built-in
DynamicSlidingWindowLayer (used by SinkCache), which solves the same problem:
the physical KV tensor is shorter than the true sequence position.

    get_seq_length()  → returns cumulative_length (true total tokens seen)
    get_mask_sizes()  → returns sizes consistent with the physical KV tensor
                        so that HF's create_causal_mask produces a mask whose
                        last dimension matches the actual KV length.

StickyCache — kept for backward compatibility.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch

try:
    from transformers.cache_utils import DynamicLayer
except ImportError:
    DynamicLayer = None  # type: ignore

try:
    from transformers import Cache
except ImportError:
    Cache = object  # type: ignore


# =========================================================================
# StickyDynamicLayer — drop-in replacement for DynamicLayer
# =========================================================================

# Inherit from DynamicLayer so HF's isinstance(layer, CacheLayerMixin)
# checks pass correctly in Cache.get_seq_length() and Cache.get_mask_sizes().
_StickyBase = DynamicLayer if DynamicLayer is not None else object


class StickyDynamicLayer(_StickyBase):
    """A cache layer that tracks true cumulative sequence length through
    eviction, following the same pattern as HF's DynamicSlidingWindowLayer.

    Key invariant:
        cumulative_length  = total tokens ever appended (grows monotonically)
        keys.shape[-2]     = physical KV length (may shrink after eviction)
        get_seq_length()   = cumulative_length  (used by HF for position_ids)
    """

    is_sliding = False
    is_compileable = False

    def __init__(self):
        if _StickyBase is not object:
            super().__init__()
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False
        self.cumulative_length = 0

    def __repr__(self):
        phys = self.keys.shape[-2] if self.is_initialized and self.keys is not None and self.keys.numel() > 0 else 0
        return f"StickyDynamicLayer(cumulative={self.cumulative_length}, physical={phys})"

    # ----- core API used by HF -----

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
        cache_kwargs: dict = None,
    ) -> tuple:
        """Standard DynamicLayer-compatible update.

        Called by the default LlamaAttention — but our STICKYLlamaAttention
        does NOT call this.  Implemented for API compatibility only.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.cumulative_length += key_states.shape[-2]
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def get_seq_length(self) -> int:
        """Return the TRUE cumulative sequence length (not compressed physical)."""
        return self.cumulative_length

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple:
        """Return mask dimensions consistent with the PHYSICAL KV tensor.

        HF 4.57 passes cache_position (a tensor), not query_length (an int).
        We derive query_length from cache_position.shape[0], matching
        DynamicLayer's exact implementation.

        Uses physical KV length (not cumulative) so the mask dimensions
        match the actual KV tensors our module works with.  Our module
        discards this mask anyway, but correct dimensions prevent crashes
        in sdpa_mask / torch.arange calls.
        """
        query_length = cache_position.shape[0]
        phys_len = 0
        if self.is_initialized and self.keys is not None and self.keys.numel() > 0:
            phys_len = self.keys.shape[-2]
        kv_length = phys_len + query_length
        kv_offset = 0
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return -1  # unbounded

    # ----- Sticky-specific API -----

    def set_evicted_kv(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Replace physical KV after eviction WITHOUT resetting cumulative_length."""
        self.keys = keys
        self.values = values
        self.is_initialized = True

    def increment_cumulative(self, n: int) -> None:
        """Increment cumulative_length (called after prefill or decode append)."""
        self.cumulative_length += n

    # ----- Standard DynamicLayer API stubs -----

    def offload(self):
        if self.is_initialized and self.keys is not None:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)

    def prefetch(self):
        pass

    def reset(self) -> None:
        if self.is_initialized and self.keys is not None:
            self.keys.zero_()
            self.values.zero_()
        self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.is_initialized and self.keys is not None and self.keys.numel() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))

    def crop(self, max_length: int) -> None:
        if self.is_initialized and self.keys is not None:
            if max_length < 0:
                max_length = self.get_seq_length() - abs(max_length)
            if self.keys.shape[-2] > max_length:
                self.keys = self.keys[..., :max_length, :]
                self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.is_initialized and self.keys is not None and self.keys.numel() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.is_initialized and self.keys is not None and self.keys.numel() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


# =========================================================================
# StickyCache — legacy wrapper (kept for backward compatibility)
# =========================================================================

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
