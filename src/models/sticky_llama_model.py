import torch
import copy
from transformers.models.llama.modeling_llama import LlamaForCausalLM

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

        print(f"DEBUG: Initializing STICKYLlamaForCausalLM with {len(self.model.layers)} layers")

        for layer_idx in range(len(self.model.layers)):
            # Explicitly overwrite the module using the ORIGINAL (unsafe) config
            self.model.layers[layer_idx].self_attn = STICKYLlamaAttention(config, layer_idx)

        print("DEBUG: All attention layers replaced.")

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for autoregressive generation.
        
        NOTE: The position_ids computed here are overridden by
        STICKYLlamaAttention.forward() which uses physical cache length for RoPE.
        This override is necessary because the KV cache is compressed by eviction,
        so the framework's position tracking is incorrect.
        
        WARNING: batch_size > 1 is structurally unsupported because
        the cache state is per-layer, not per-batch-item.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        if past_key_values is not None:
            # Override incorrect slicing done by super() due to evicted cache size.
            # After eviction the physical cache is shorter than the true sequence,
            # so super() may feed too many tokens. Force single-token decode.
            model_inputs["input_ids"] = input_ids[:, -1:]
            
            # Position IDs generation needs to account for the total generated length
            # because the KV cache has been artificially shortened by eviction.
            # `input_ids.shape[1]` perfectly tracks the true global sequence length
            # during transformers `.generate()` loops.
            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    model_inputs["position_ids"] = position_ids[:, -1:]
                else:
                    true_seq_length = input_ids.shape[1]
                    model_inputs["position_ids"] = torch.tensor([[true_seq_length - 1]], dtype=torch.long, device=input_ids.device)
            else:
                model_inputs["position_ids"] = position_ids[:, -1:]
        
        return model_inputs