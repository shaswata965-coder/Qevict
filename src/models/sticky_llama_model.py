import torch
import copy
from transformers.models.llama.modeling_llama import LlamaForCausalLM
try:
    from src.models.sticky_cache import StickyCache
except ImportError:
    from .sticky_cache import StickyCache

# Backend selection is controlled by USE_FLASH_ATTENTION in sticky_config.
# tracking_flag only controls whether the TrackingManager is active inside
# the cache; it does not change the attention compute backend.
#
#   USE_FLASH_ATTENTION=1 -> Flash Attention v2 backend (module_flash.py)
#   USE_FLASH_ATTENTION=0 -> Standard SDPA backend (module.py), default
#
# The SDPA backend works on any GPU without flash_attn installed and is
# required for the run_longbench_sticky evaluation path.
try:
    from src.sticky_config import USE_FLASH_ATTENTION as _use_fa
except (ImportError, AttributeError):
    try:
        from sticky_config import USE_FLASH_ATTENTION as _use_fa
    except (ImportError, AttributeError):
        _use_fa = 0

if _use_fa == 1:
    from src.models.sticky_llama_attention_fast_attention import STICKYLlamaAttention
else:
    from src.models.sticky_llama_attention import STICKYLlamaAttention


class STICKYLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, **kwargs):
        # Match originalQevict exactly: initialize the parent HF Llama stack with
        # rope_scaling stripped, then install our custom attention layers with
        # the original config. This keeps HF's internal rotary path out of the
        # parent construction while preserving Llama 3 RoPE in STICKYLlamaAttention.
        safe_config = copy.deepcopy(config)
        if hasattr(safe_config, "rope_scaling"):
            safe_config.rope_scaling = None
        safe_config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": getattr(safe_config, "rope_theta", 10000.0),
        }

        super().__init__(safe_config)

        self.config = config

        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].self_attn = STICKYLlamaAttention(config, layer_idx)

        print(f"Loaded STICKYLlamaForCausalLM - {len(self.model.layers)} layers, backend={'flash' if _use_fa else 'sdpa'}")

    # NOTE: _get_cache() is intentionally NOT overridden.
    #
    # Many HF versions never call _get_cache(). Instead, LlamaModel.forward()
    # creates a DynamicCache directly when past_key_values is None.
    # Our attention module works with ANY cache type (DynamicCache, StickyCache)
    # by reading/writing via the cache.key_cache / cache.value_cache lists.
    #
    # If _get_cache IS called by a particular HF version, DynamicCache works
    # fine; our module updates it in place.

    def _get_true_global_position(self):
        """Read global_token_counter from layer 0's kv_cache.

        Returns the TRUE number of tokens seen so far (not the compressed
        physical cache length), or None if unavailable.
        """
        try:
            layer0_attn = self.model.layers[0].self_attn
            if hasattr(layer0_attn, "kv_cache"):
                return int(layer0_attn.kv_cache.global_token_counter.item())
        except (IndexError, AttributeError):
            pass
        return None

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for autoregressive generation.

        After eviction compresses the cache, HF's super() method can compute
        cache_position and position_ids from the physical cache length. Those
        are wrong for Sticky KV because RoPE and causal alignment need the true
        global token position, so we overwrite them when a cache is present.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if past_key_values is not None:
            device = input_ids.device
            prep_dbg_count = getattr(self, "_dbg_prepare_count", 0)

            # Force single-token decode regardless of how super() sliced input_ids.
            # The physical cache is compressed by eviction, so HF may otherwise
            # think fewer tokens have been processed and feed multiple tokens.
            model_inputs["input_ids"] = input_ids[:, -1:]

            # Ensure the cache object flows through.
            model_inputs["past_key_values"] = past_key_values
            model_inputs["use_cache"] = True

            global_tc = self._get_true_global_position()
            if global_tc is not None:
                model_inputs["cache_position"] = torch.tensor(
                    [global_tc], dtype=torch.long, device=device
                )
                model_inputs["position_ids"] = torch.tensor(
                    [[global_tc]], dtype=torch.long, device=device
                )

            if prep_dbg_count < 10:
                cache_pos = model_inputs.get("cache_position", None)
                pos_ids = model_inputs.get("position_ids", None)
                cache_pos_dbg = cache_pos.detach().flatten().tolist() if torch.is_tensor(cache_pos) else cache_pos
                pos_ids_dbg = pos_ids.detach().flatten().tolist() if torch.is_tensor(pos_ids) else pos_ids
                print(
                    f"[GEN-PREP current step={prep_dbg_count}] "
                    f"input_len={input_ids.shape[1]} sliced_len={model_inputs['input_ids'].shape[1]} "
                    f"global_tc={global_tc} cache_position={cache_pos_dbg} position_ids={pos_ids_dbg} "
                    f"pkv_type={type(past_key_values).__name__}",
                    flush=True,
                )
            self._dbg_prepare_count = prep_dbg_count + 1

        return model_inputs
