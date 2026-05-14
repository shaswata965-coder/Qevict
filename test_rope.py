import torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
rope = LlamaRotaryEmbedding(config)

x = torch.randn(1, 8, 10, 64)
pos = torch.arange(10).unsqueeze(0)
cos, sin = rope(x, pos)

print("cos shape:", cos.shape)
print("sin shape:", sin.shape)
