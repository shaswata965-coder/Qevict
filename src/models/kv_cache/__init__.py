"""
kv_cache/__init__.py
--------------------
Public API for the kv_cache sub-package.

Re-exports every symbol that was previously importable from
``src.models.sticky_kv_logic_cummulative`` so that all existing call sites
continue to work with zero modifications.

Typical usage (unchanged from before):

    from src.models.kv_cache import STICKYKVCache_LayerWise
    from src.models.kv_cache import repeat_kv, _make_causal_mask

Manager classes are also exported for testing and debug inspection:

    from src.models.kv_cache import EvictionManager
    from src.models.kv_cache import QuantizationManager, NoOpQuantizationManager
    from src.models.kv_cache import TrackingManager, NoOpTrackingManager
"""

from .cache import STICKYKVCache_LayerWise
from .helpers import repeat_kv, _make_causal_mask, apply_rotary_pos_emb_single

from .base_quantization_manager import BaseQuantizationManager
from .base_tracking_manager import BaseTrackingManager
from .eviction_manager import EvictionManager
from .quantize_manager import QuantizationManager, NoOpQuantizationManager
from .tracking_manager import TrackingManager, NoOpTrackingManager

__all__ = [
    # Core
    "STICKYKVCache_LayerWise",
    "repeat_kv",
    "_make_causal_mask",
    "apply_rotary_pos_emb_single",
    # ABCs
    "BaseQuantizationManager",
    "BaseTrackingManager",
    # Managers
    "EvictionManager",
    "QuantizationManager",
    "NoOpQuantizationManager",
    "TrackingManager",
    "NoOpTrackingManager",
]
