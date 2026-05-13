# Sticky KV Cache — Debugging Handoff

## Current Status

**Score: 0.1085** (LCC, seed=42, 20 samples)  
**Expected: 0.3265** (from `originalQevict/` running the same config)  
**Gap: 3× regression — deterministic, not noise**

The modularized code runs without crashes but produces severely degraded output quality.

---

## Architecture Overview

The original monolithic `originalQevict/sticky_kv_logic_cummulative.py` (~1400 lines) was split into:

| Module | Purpose |
|---|---|
| `src/models/kv_cache/cache.py` | `STICKYKVCache_LayerWise` coordinator — owns the `__call__` lifecycle |
| `src/models/kv_cache/eviction_manager.py` | Window scoring, decode eviction cycle, logical_id_map |
| `src/models/kv_cache/quantize_manager.py` | Q-cache storage, rebuild, dequantize |
| `src/models/kv_cache/tracking_manager.py` | Token ledger, global_token_counter (NoOp when tracking_flag=0) |
| `src/models/kv_cache/allocator.py` | CFO budget allocator (`compute_budget`) |
| `src/models/kv_cache/eviction.py` | Pure eviction logic (`evict_from_window_scores`) |
| `src/models/attention/module.py` | `STICKYLlamaAttention` — SDPA forward pass |
| `src/models/attention/ops.py` | Attention math (logits, softmax, q-cache joint softmax) |
| `src/models/attention/rope.py` | `Llama3RotaryEmbedding`, `HFRopeWrapper`, `init_rope()` |
| `src/models/sticky_cache.py` | `StickyCache(Cache)` — HF Cache subclass for generate() loop |
| `src/models/sticky_llama_model.py` | `STICKYLlamaForCausalLM` — model wrapper |

The original `originalQevict/` code is preserved in the repo for reference comparison.

---

## Configuration (Verified Identical)

Both `originalQevict/sticky_config.py` and `src/sticky_config.py` use:

```
R_RATIO = 20
P_RATIO = 50
Q_RATIO = 10
QUANTIZATION_BIT_WIDTH = 4
OMEGA = 8
SINK_TOKENS = 4
tracking_flag = 0
do_sample = False
max_new_tokens = 512
```

**One difference found but deemed non-causal:** The original `main.py` sets `sticky_config.LOCAL_NUM_TOKENS = 256` at runtime (fixed local tokens), while our code uses `P_RATIO = 50` (percentage-based). Both give similar local token counts for typical prompt lengths. `LOCAL_NUM_TOKENS` is commented out in our `sticky_config.py`.

---

## What Was Already Fixed (Runtime Crashes)

### 1. Mapping tensor unconditional allocation (cache.py)
**Problem:** `mapping` tensor was always created and populated even with `tracking_flag=0`. Original code made it conditional: `mapping = ... if self.tracking_flag else None`.  
**Fix:** Made `mapping` conditional on `self._tracking_flag`. All 5 write sites (`sink`, `sticky windows`, `promoted Q-cache`, `fallback windows`, `local zone`) now guarded with `if mapping is not None`.

### 2. Q-cache promoted window mapping write (cache.py)
**Problem:** Mapping write for promoted Q-cache windows was suppressed with a comment about rerotation. But rerotation was removed, so the mapping write should exist (for tracking parity).  
**Fix:** Restored `mapping` write for promoted Q-cache windows, matching `originalQevict` lines 1083-1091.

### 3. Path C bounds check (quantize_manager.py line 248)
**Problem:** `v_fp = past_key_values[1][0, h_val, ps:ps + omega]` returned 0 tokens when `ps + omega > seq_len`, causing shape mismatch with archived `vz` tensor `[omega, 1]`.  
**Fix:** Added `if ps + omega <= seq_len` guard; falls back to zero-fill when out of bounds.

### 4. CUDA index OOB in `_build_physical_cache` (cache.py line 480)
**Problem:** `first_phys + omega` could exceed `seq_len` with tight R_RATIO=20 budgets. Vectorized gather read past tensor bounds.  
**Fix:** Added `in_bounds = (first_phys + omega) <= seq_len; found_in_main = found_in_main & in_bounds` to both SDPA and FA2 cache.py files + `quantize_manager.py` Path B.

### 5. `.max()` vs `.min()` for `valid_old_windows` (eviction_manager.py line 261)
**Problem:** Changed from original `.min()` to `.max()` during modularization with incorrect "WARN-2 fix" comment. Using `.max()` could read NaN-padded slots for heads with fewer valid windows.  
**Fix:** Restored `.min()` to match original `FIX (C1)`.  
**Note:** This fix was applied but the score did NOT change (still 0.1085), suggesting this is NOT the root cause for this particular test configuration, or the fix wasn't deployed.

---

## What Was Already Verified (Logic Parity)

### Line-by-line confirmed identical:

1. **`compute_budget` vs `_update_k_win_and_local_num`** — Sequential carving allocator is identical. Tiny `round()` vs `int()` difference in `q_int4_target` (~1 token).

2. **`tally_prefill_scores`** — Window scoring from attention during prefill. Same slice logic, same `window_scores` tensor population, same `local_history` seeding.

3. **`run_decode_cycle`** — Scoreboard reconstruction, challenger scoring, top-k/top-q selection, `already_tracked_per_head` dedup guard. All match original lines 560-780.

4. **`_build_physical_cache`** — Physical cache reconstruction: sink copy, sticky window vectorized batch copy, Q-cache promoted window copy, fallback span lookup, local zone copy. All match original lines 990-1110 (with bounds guards added).

5. **`quantize_manager.rebuild`** — Path A (retained from old Q-cache), Path B (batch gather + quantize), Path C (archived meta re-quantize). All match original lines 870-960.

6. **`Llama3RotaryEmbedding`** — Identical YaRN-scaled frequency computation. Same `_set_cos_sin_cache`, same `forward(x, seq_len)` API.

7. **`apply_rotary_pos_emb_single`** — Imported from `sticky_kv_logic_cummulative.py` directly.

8. **`_make_causal_mask`** — Same function imported directly.

9. **`compute_main_logits`** — GQA reshape + matmul, identical to original lines 206-214.

10. **`compute_qcache_joint_softmax`** — Dequantize, q_logits, joint softmax, split weights, GQA output. Identical to original lines 225-283.

11. **`compute_standard_softmax`** — Standard softmax path. Identical to original lines 288-305.

12. **`apply_prefill_causal_mask`** — In-place causal masking. Identical to original lines 218-223.

13. **Position IDs during decode** — Both use `global_token_counter` to compute true sequence position, not compressed cache length.

14. **Position IDs during prefill** — Both use `[0, 1, ..., q_len-1]`.

15. **TrackingManager state update timing** — The timing of `global_token_counter` increments (happens at the very start of `__call__` for both) and ledger tracking updates (happens before the `tokens_since_last_review == omega` eviction check) was fully audited and perfectly matches the original's sequence.

16. **`engine.py` `generate()`** — Cache cleanup, gen_kwargs, model.generate() call, output decoding. Identical between original and new.

17. **Delegating properties on `STICKYKVCache_LayerWise`** — `q_cache_k_quant`, `q_cache_k_scale`, `q_cache_k_zp`, etc. all delegate to `QuantizationManager` correctly. `hasattr()` returns True.

18. **`prepare_inputs_for_generation` (input slicing)** — Both force single-token decode via `input_ids[:, -1:]`.

### Rerotation removal (design decision):
The original stored keys with absolute RoPE positions. During a previous conversation, rerotation logic was added (DefensiveKV-style) and then removed again. The current state: **keys are stored with their original absolute RoPE embeddings** — no un-rotation or re-rotation at any point. This matches the original's behavior.

---

## What Has NOT Been Verified

### 1. Whether `_get_cache()` is actually called by HF's `generate()`

Our `STICKYLlamaForCausalLM._get_cache()` is overridden to return a `StickyCache`. But some HF versions may not call this method. If it's not called, HF's `LlamaModel.forward()` creates a `DynamicCache` internally (line 889-890 in v4.46). Our `module.py` has a fallback path for DynamicCache (lines 107-113), but `DynamicCache._seen_tokens` is never updated, which could cause HF to compute wrong `cache_position` and wrong causal masks.

### 2. Whether HF pre-computed `position_embeddings` interfere

In HF v4.46+, `LlamaModel.forward()` computes `position_embeddings = self.rotary_emb(hidden_states, position_ids)` and passes them to each decoder layer. Our module captures this in `**kwargs` and ignores it, computing our own RoPE. This should be fine, but needs verification that no downstream HF code uses the pre-computed embeddings.

### 3. Whether Q/K/V/O weights are loaded correctly

`STICKYLlamaForCausalLM.__init__` creates standard LlamaAttention modules via `super().__init__()`, then REPLACES them with `STICKYLlamaAttention`. The `from_pretrained()` state dict loading happens after. Both use the same submodule names (`q_proj`, `k_proj`, `v_proj`, `o_proj`), so keys should match. But this hasn't been empirically verified on the Kaggle environment.

### 4. Attention Return Signature (2-value vs 3-value)

The original `STICKYLlamaAttention.forward()` returned **3 values**: `(attn_output, attn_weights, past_key_value)`. 
Our modularized `module.py` returns **2 values**: `(attn_output, attn_weights)`. 
While this aligns with HF v4.46+ which extracts `hidden_states, _ = self.self_attn(...)`, if the Kaggle HF version expects the cache as a third return value, it might be silently discarding our cache or failing to propagate it properly if the `StickyCache` in-place update fails.

### 5. `prepare_inputs_for_generation` Position IDs Override

The original `sticky_llama_model.py` manually calculated `position_ids` inside `prepare_inputs_for_generation` based on the attention mask or input length. Our modularized version intentionally omits this, deferring entirely to `module.py`'s `global_token_counter`. If HF uses the model-level `position_ids` for anything other than the RoPE calculation (which we safely override), it could cause issues.

### 6. The exact HF transformers version on Kaggle

The error tracebacks show `modeling_llama.py line 320` for `LlamaDecoderLayer` and `line 421` for `LlamaModel` — these don't match v4.46.0 line numbers. The actual version determines which code paths are active.

### 7. Whether the `DynamicCache` fallback path actually works end-to-end

If `_get_cache` is not called and HF uses DynamicCache, our code reads/writes via `key_cache[layer_idx]` list manipulation. But DynamicCache may have internal invariants that we violate (e.g., `_seen_tokens` never updated, `update()` never called). If `_seen_tokens` stays at 0, HF might generate an incorrect global causal mask in `_update_causal_mask` which is then passed to our layer.

---

## Diagnostic Prints Added (Ready to Deploy)

### File: `src/models/sticky_llama_model.py`

```python
# In _get_cache():
print(f"[DBG] _get_cache called — returning StickyCache(num_layers={num_layers})", flush=True)

# In prepare_inputs_for_generation() (first 3 decode calls):
print(f"[DBG prep_inputs #{count}] cache={cache_type} input_ids_shape={inp_shape} position_ids={pos_ids}", flush=True)
```

### File: `src/models/attention/module.py`

Layer 0 only, first 5 steps per sample:

```python
# After cache extraction:
print(f"[DBG L0 step={n}] q_len={q_len} cache_type={cache_type} past_kv_shape={past_kv_shape}", flush=True)

# After RoPE + concatenation:
print(f"[DBG L0 step={n}] pos_ids={pos_ids[0,:3].tolist()}... global_tc={global_token_counter} phys_past={phys_past_len} has_qcache={has_qcache} concat_k_shape={key_states.shape}", flush=True)
```

### What the Output Reveals

| Output Pattern | Diagnosis |
|---|---|
| `_get_cache` NOT printed | HF doesn't call our override → StickyCache never created |
| `cache_type=DynamicCache` | HF is using DynamicCache → our fallback path |
| `cache_type=StickyCache` | StickyCache IS being used → issue is elsewhere |
| `past_kv_shape=None` on decode | Cache not flowing between steps |
| `global_tc=0` on decode | global_token_counter not incrementing |
| `global_tc=4096` on first decode | Counter correct ✓ |
| `has_qcache=False` always | Q-cache never activates (check after omega*k decode steps) |
| `has_qcache=True` after ~8 decode steps | Q-cache working ✓ |
| `pos_ids=[0]` on decode | Position IDs wrong (should be [4096], [4097], etc.) |
| `phys_past=0` on decode | Cache extraction failed |

### Expected Correct Output

```
[DBG] _get_cache called — returning StickyCache(num_layers=16)
[DBG L0 step=0] q_len=~4000 cache_type=StickyCache past_kv_shape=None
[DBG L0 step=0] pos_ids=[0, 1, 2]... global_tc=~4000 phys_past=0 has_qcache=False concat_k_shape=[1, 8, ~4000, 64]
[DBG prep_inputs #0] cache=StickyCache input_ids_shape=torch.Size([1, 1]) position_ids=...
[DBG L0 step=1] q_len=1 cache_type=StickyCache past_kv_shape=torch.Size([1, 8, ~800, 64])
[DBG L0 step=1] pos_ids=[~4000]... global_tc=~4000 phys_past=~800 has_qcache=False concat_k_shape=[1, 8, ~801, 64]
```

---

## Files Modified in This Conversation

| File | Changes |
|---|---|
| `src/models/kv_cache/cache.py` | Conditional mapping, in_bounds guard, promoted Q-cache mapping restore |
| `src/models/kv_cache/quantize_manager.py` | Path C bounds check, Path B in_bounds guard |
| `src/models/kv_cache/eviction_manager.py` | `.max()` → `.min()` fix |
| `src/models/kv_cache_fast_attention/cache.py` | in_bounds guard (FA2 parity) |
| `src/models/attention/module.py` | Diagnostic prints (layer 0, first 5 steps) |
| `src/models/sticky_llama_model.py` | Diagnostic prints in `_get_cache` and `prepare_inputs_for_generation` |

---

## Key Conversation IDs for Context

| Conversation | Topic |
|---|---|
| `9eeeb5bc-4376-43cb-8904-b68e78813de2` | **Current** — Bounds fixes + deep analysis |
| `f4076259-e2d8-4d87-843c-a72271cedb6b` | DynamicCache bridge, absolute RoPE, autoregressive prep |
| `5829662c-95bb-4548-8fcb-cedbd97d0857` | Rotary embedding dimension mismatch, multi-GPU sync |
| `2c83889d-5648-4146-b3a5-74e18a85aa22` | DefensiveKV RoPE rerotation integration (later removed) |

---

## Next Steps

1. **Run evaluation with diagnostic prints** — read the `[DBG]` output
2. **If `_get_cache` not called** → The fix is to ensure StickyCache is created. Options:
   - Override `forward()` at the `LlamaForCausalLM` level to inject StickyCache
   - Or set `past_key_values = StickyCache(...)` in `prepare_inputs_for_generation` when `past_key_values is None`
3. **If cache_type=DynamicCache** → HF is wrapping our cache. Fix by ensuring `isinstance(StickyCache, Cache)` passes
4. **If everything looks correct in diagnostics** → The issue may be in weight loading. Add a print showing `model.model.layers[0].self_attn.q_proj.weight[:3,:3]` before and after `from_pretrained` to verify weights loaded
5. **After root cause is fixed** → Remove all `[DBG]` prints and re-run clean evaluation
