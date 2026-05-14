"""
attention/rope.py
-----------------
Rotary positional embedding for Llama 3 (YaRN-scaled) and standard Llama fallback.

Shared by both the standard SDPA backend (module.py) and the Flash-Attention
backend (module_flash.py).  No logic changes from the original
sticky_llama_attention.py – only extracted into a standalone module.
"""

import math
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


# ---------------------------------------------------------------------------
# Llama 3 YaRN-scaled rotary embedding
# ---------------------------------------------------------------------------

class Llama3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=None,
        base=None,
        device=None,
        scaling_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        original_max_position_embeddings=None,
    ):
        super().__init__()
        try:
            import src.sticky_config as sticky_config
        except ImportError:
            import sticky_config  # fallback: running from inside src/
        max_position_embeddings = max_position_embeddings if max_position_embeddings is not None else sticky_config.MAX_POSITION_EMBEDDINGS
        base = base if base is not None else sticky_config.ROPE_THETA
        scaling_factor = scaling_factor if scaling_factor is not None else sticky_config.ROPE_SCALING_FACTOR
        low_freq_factor = low_freq_factor if low_freq_factor is not None else sticky_config.ROPE_LOW_FREQ_FACTOR
        high_freq_factor = high_freq_factor if high_freq_factor is not None else sticky_config.ROPE_HIGH_FREQ_FACTOR

        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        if original_max_position_embeddings is None:
            original_max_position_embeddings = sticky_config.ORIGINAL_MAX_POSITION_EMBEDDINGS
        self.original_max_position_embeddings = original_max_position_embeddings

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _recompute_inv_freq(self, device):
        """Recompute inv_freq from scalar hyperparameters.

        inv_freq is a persistent=False buffer.  When a model is loaded with
        ``device_map="auto"``, HF Accelerate materialises the buffer on the
        target device via ``torch.empty_like``, leaving it filled with
        uninitialised (often zero) memory.  This method regenerates it from
        the stored scalar hyperparameters (self.base, self.dim), which are
        plain Python numbers and are always correct after __init__.
        """
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        # Guard: inv_freq is a persistent=False buffer and may be garbage after
        # an accelerate meta-device load.  If it looks invalid (all zeros or
        # wrong device), regenerate it from hyperparameters before use.
        if (
            not hasattr(self, "inv_freq")
            or self.inv_freq.device != torch.device(device)
            or self.inv_freq.abs().sum().item() == 0.0
        ):
            self._recompute_inv_freq(device)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        inv_freq = self.inv_freq

        # Calculate wavelengths
        wavelen = 2 * math.pi / inv_freq

        # Create smooth factor
        smooth = (self.original_max_position_embeddings / wavelen - self.low_freq_factor) / (
            self.high_freq_factor - self.low_freq_factor
        )
        smooth = torch.clamp(smooth, 0, 1)

        # Apply scaling
        scaled_inv_freq = inv_freq / self.scaling_factor

        # Correct logic from Llama 3 reference:
        # new_freqs = (1 - smooth) * scaled_inv_freq + smooth * inv_freq
        new_inv_freq = (1 - smooth) * scaled_inv_freq + smooth * inv_freq

        freqs = torch.outer(t, new_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # FIX: When models are loaded with device_map="auto", HF Accelerate creates the
        # model on the 'meta' device.  persistent=False buffers (cos_cached, sin_cached,
        # and inv_freq) are moved to the GPU as uninitialised memory (often zeros)
        # instead of running the initialisation computation.
        #
        # Strategy:
        #   1. On the very first real forward pass (_is_initialized is False), force a
        #      full recompute.  _set_cos_sin_cache internally validates inv_freq and
        #      regenerates it from scalar hyperparameters if it looks stale.
        #   2. When the sequence grows beyond the cached length, extend the cache
        #      (standard behaviour, also guarded by the same inv_freq validation).
        #   3. _is_initialized is set only AFTER a successful recompute so a failed
        #      compute doesn't silently skip future attempts.
        if not getattr(self, "_is_initialized", False) or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=max(seq_len, self.max_seq_len_cached),
                device=x.device,
                dtype=x.dtype,
            )
            self._is_initialized = True  # set only after successful recompute

        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        )


class HFRopeWrapper(nn.Module):
    """
    Wraps the HuggingFace LlamaRotaryEmbedding to provide a consistent `forward(x, seq_len=...)` 
    API, regardless of whether the underlying HF version expects `seq_len` (older versions) 
    or `position_ids` (modern versions >=4.43+).
    
    The API style is detected once at init time to avoid the overhead of 
    inspect.signature() on every forward call during generation.
    """
    def __init__(self, hf_rope):
        super().__init__()
        self.hf_rope = hf_rope
        
        # Detect API style once at construction time
        import inspect
        sig = inspect.signature(hf_rope.forward)
        self._uses_position_ids = "position_ids" in sig.parameters

    def forward(self, x, seq_len=None):
        if self._uses_position_ids:
            # Modern HF uses position_ids
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0)  # [1, seq_len]
            
            cos, sin = self.hf_rope(x, position_ids)
            # Modern HF returns [batch, seq_len, head_dim] -> squeeze batch to [seq_len, head_dim]
            return cos.squeeze(0), sin.squeeze(0)
        else:
            # Legacy HF uses seq_len
            return self.hf_rope(x, seq_len)

def init_rope(config, head_dim: int, max_position_embeddings: int, rope_theta: float) -> nn.Module:
    """Build and return the appropriate rotary embedding module for a given config.

    Extracted verbatim from STICKYLlamaAttention._init_rope so that both the
    standard and flash-attention backends share identical initialisation logic.
    """
    # Check for Llama 3 specific config
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None and isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
        if rope_type == "llama3":
            dim = head_dim
            max_pos = max_position_embeddings
            base = rope_theta

            try:
                import src.sticky_config as sticky_config
            except ImportError:
                import sticky_config  # fallback: running from inside src/
            factor = rope_scaling.get("factor", sticky_config.ROPE_SCALING_FACTOR)
            low_freq = rope_scaling.get("low_freq_factor", sticky_config.ROPE_LOW_FREQ_FACTOR)
            high_freq = rope_scaling.get("high_freq_factor", sticky_config.ROPE_HIGH_FREQ_FACTOR)
            orig_max_pos = rope_scaling.get(
                "original_max_position_embeddings",
                sticky_config.ORIGINAL_MAX_POSITION_EMBEDDINGS,
            )

            return Llama3RotaryEmbedding(
                dim=dim,
                max_position_embeddings=max_pos,
                base=base,
                scaling_factor=factor,
                low_freq_factor=low_freq,
                high_freq_factor=high_freq,
                original_max_position_embeddings=orig_max_pos,
            )

    # Fallback to standard HuggingFace LlamaRotaryEmbedding
    try:
        hf_rope = LlamaRotaryEmbedding(config)
        return HFRopeWrapper(hf_rope)
    except (TypeError, AttributeError):
        dim = head_dim
        max_pos = getattr(config, "max_position_embeddings", 2048)
        base = getattr(config, "rope_theta", 10000.0)

        try:
            hf_rope = LlamaRotaryEmbedding(dim, max_pos, base=base)
            return HFRopeWrapper(hf_rope)
        except Exception:
            hf_rope = LlamaRotaryEmbedding(dim, max_position_embeddings=max_pos)
            hf_rope.base = base
            return HFRopeWrapper(hf_rope)
