"""
kv_cache/ledger.py
------------------
Token ledger accessor for the Sticky KV-cache.

Extracted verbatim from STICKYKVCache_LayerWise.get_ledger_data()
(sticky_kv_logic_cummulative.py, lines 1165–1179).

Narrow API: only the three tensors the function actually reads are passed in.
No logic is added, removed, or reordered.
"""

import torch


def get_ledger_data(global_token_counter: torch.Tensor,
                    token_ledger: torch.Tensor,
                    num_heads: int) -> dict:
    """Return a structured view of the active portion of the token ledger.

    Parameters
    ----------
    global_token_counter : scalar LongTensor  — total tokens processed so far.
    token_ledger         : [max_context, 2 + 2*num_heads] float32 tensor.
    num_heads            : number of KV heads.

    Returns
    -------
    dict with keys: global_id, layer_id, physical_id, attention_score.
    """
    total_processed = global_token_counter.item()
    active_ledger = token_ledger[:total_processed].detach().cpu()
    
    global_ids = active_ledger[:, 0].long()
    layer_ids = active_ledger[:, 1].long()
    physical_positions = active_ledger[:, 2:2+num_heads].long()
    attention_scores = active_ledger[:, 2+num_heads:2+2*num_heads]
    
    return {
        "global_id": global_ids,
        "layer_id": layer_ids,
        "physical_id": physical_positions,
        "attention_score": attention_scores 
    }
