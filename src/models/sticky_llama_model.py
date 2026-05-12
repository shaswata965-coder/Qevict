import torch
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
        super().__init__(config)
        self.config = config
        
        print(f"DEBUG: Initializing STICKYLlamaForCausalLM with {len(self.model.layers)} layers")
        
        for layer_idx in range(len(self.model.layers)):
            # Explicitly overwrite the module using the ORIGINAL (unsafe) config
            self.model.layers[layer_idx].self_attn = STICKYLlamaAttention(config, layer_idx)
        
        print("DEBUG: All attention layers replaced.")
