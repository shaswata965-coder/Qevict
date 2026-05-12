import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# Select attention backend based on tracking_flag:
#   tracking_flag=1 → SDPA (cumulative) backend with full tracking support
#   tracking_flag=0 → Flash Attention backend (no per-token ledger)
# Eval scripts (run_sticky_baseline_cummulative, run_open_ended_sticky_baseline)
# require the cumulative backend for token_ledger and prefill_attention_matrix access.
try:
    from src.sticky_config import tracking_flag as _tracking_flag
except ImportError:
    try:
        from sticky_config import tracking_flag as _tracking_flag
    except ImportError:
        _tracking_flag = 1  # default: tracking enabled → SDPA backend

if _tracking_flag == 1:
    from src.models.sticky_llama_attention import STICKYLlamaAttention
else:
    from src.models.sticky_llama_attention_fast_attention import STICKYLlamaAttention

class STICKYLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        
        print(f"DEBUG: Initializing STICKYLlamaForCausalLM with {len(self.model.layers)} layers")
        
        for layer_idx in range(len(self.model.layers)):
            # Explicitly overwrite the module using the ORIGINAL (unsafe) config
            self.model.layers[layer_idx].self_attn = STICKYLlamaAttention(config, layer_idx)
        
        print("DEBUG: All attention layers replaced.")
