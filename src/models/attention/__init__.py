"""
attention/__init__.py
---------------------
Public API for the attention sub-package.

Importing from ``src.models.attention`` gives access to the same symbols
that were previously exported from the flat sticky_llama_attention.py file,
so any code using:

    from src.models.attention import STICKYLlamaAttention

works without modification.  The flash-attention variant must be imported
explicitly from attention.module_flash.
"""

from .module import STICKYLlamaAttention          # standard SDPA backend
from .rope import Llama3RotaryEmbedding, init_rope

__all__ = [
    "STICKYLlamaAttention",
    "Llama3RotaryEmbedding",
    "init_rope",
]
