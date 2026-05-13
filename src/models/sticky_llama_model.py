import torch
import copy
from transformers.models.llama.modeling_llama import LlamaForCausalLM
try:
    from src.models.sticky_cache import StickyCache
except ImportError:
    from .sticky_cache import StickyCache

# Backend selection is controlled by USE_FLASH_ATTENTION in sticky_config.
# tracking_flag only controls whether the TrackingManager is active inside
# the cache — it does NOT change the attention compute backend.
#
#   USE_FLASH_ATTENTION=1 → Flash Attention v2 backend (module_flash.py)
#   USE_FLASH_ATTENTION=0 → Standard SDPA backend (module.py)  ← default
#
# The SDPA backend works on any GPU without flash_attn installed and is
# required for the run_longbench_sticky evaluation path.
try:
    from src.sticky_config import USE_FLASH_ATTENTION as _use_fa
except (ImportError, AttributeError):
    try:
        from sticky_config import USE_FLASH_ATTENTION as _use_fa
    except (ImportError, AttributeError):
        _use_fa = 0  # default: SDPA backend

if _use_fa == 1:
    from src.models.sticky_llama_attention_fast_attention import STICKYLlamaAttention
else:
    from src.models.sticky_llama_attention import STICKYLlamaAttention

class STICKYLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, **kwargs):
        # Modern transformers (>= 4.43) handles "llama3" rope_type natively.
        # For older transformers that crash on unknown rope types, we sanitize
        # rope_scaling BEFORE calling super().__init__() so we never call it twice.
        # Calling super().__init__() twice on a partially-initialised object causes
        # self.model.layers to be None → 'NoneType' object is not subscriptable.
        rope_scaling_backup = getattr(config, "rope_scaling", None)

        # Probe whether super().__init__ will accept this rope_scaling without
        # actually running it — we do a lightweight structural check instead.
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type", "")
            allowed = {"linear", "dynamic", "llama3", "yarn", "longrope", ""}
            if rope_type not in allowed:
                # Unknown type — clear it now so super().__init__ won't crash
                config.rope_scaling = None

        try:
            super().__init__(config)
        except (ValueError, TypeError, KeyError) as e:
            # Last-resort fallback: strip rope_scaling and retry.
            # Only reached if the structural check above missed something.
            import warnings
            warnings.warn(
                f"STICKYLlamaForCausalLM: super().__init__ failed ({e}). "
                "Retrying after clearing rope_scaling — this should not normally happen."
            )
            # We must NOT call super().__init__() again on the same object.
            # Create a fresh config copy for the retry instead.
            config_copy = copy.deepcopy(config)
            config_copy.rope_scaling = None
            super().__init__(config_copy)

        # Always restore the original rope_scaling on the live config object
        config.rope_scaling = rope_scaling_backup
        self.config = config

        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].self_attn = STICKYLlamaAttention(config, layer_idx)

        print(f"Loaded STICKYLlamaForCausalLM — {len(self.model.layers)} layers, backend={'flash' if _use_fa else 'sdpa'}")

    def _get_cache(self, *args, **kwargs):
        """Override: always use StickyCache so our attention gets its typed slot API.

        HF's generate() calls this (with varying signatures across versions) to
        create the initial cache object before the first forward pass.
        Accepting *args/**kwargs makes this forward-compatible with any HF version.
        """
        num_layers = self.config.num_hidden_layers
        return StickyCache(num_layers=num_layers)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for autoregressive generation.

        We call super() and then forcibly fix the input_ids slice:
        after eviction the physical cache is shorter than the true sequence,
        so super() may feed too many tokens — we always force single-token decode.

        position_ids are intentionally NOT overridden here. They are computed
        correctly inside STICKYLlamaAttention.forward() using global_token_counter
        (the true sequence position), which is more accurate than anything we
        can compute from the compressed cache length here.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, **kwargs
        )
        if past_key_values is not None:
            # Force single-token decode regardless of how super() sliced input_ids
            model_inputs["input_ids"] = input_ids[:, -1:]

        return model_inputs