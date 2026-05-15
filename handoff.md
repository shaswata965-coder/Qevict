# Sticky KV Cache Debugging Handoff

Last updated: 2026-05-15

## Executive Summary

We are debugging why the modularized `src/` implementation of Sticky KV gives much worse LCC quality than `originalQevict/`, despite the sticky eviction/scoring code appearing logically equivalent.

The current observed bad runs:

- Dataset/task: `lcc`
- Seed: `42`
- Sample count: `20`
- Current score shown in recent diagnostics: about `0.1260 +/- 0.0627` in one run, with sample 2 scoring `0.0000`
- Another current run in `debug.log`: about `0.1715 +/- 0.0788`
- Earlier low scores observed during this debugging sequence: `0.1085`, then `0.04` after cache propagation began working
- Expected/reference score from `originalQevict/`: about `0.3265`

Important: we have **not** proven that the Hugging Face cache contract is the only cause. It is a real and repeatedly observed difference, but other causes remain plausible. The step-1 decode cache comparison has now been run, and it rules out one earlier hypothesis: originalQevict is **not** scoring higher because it enters the first decode token with a larger/full physical cache.

The current strongest hypothesis is:

> The sticky scoring/eviction math may be equivalent at the first decode token, but the effective generation protocol can still diverge after step 1. The remaining likely causes are decode-time cache lifecycle differences: `DynamicCache` in-place writeback versus original tuple-return cache, `cache_position` handling, q-cache/review-cycle behavior after `OMEGA` tokens, and possibly a still-hidden tensor-content difference in retained windows.

## Repository Layout Relevant To This Investigation

`originalQevict/` is preserved as the monolithic reference implementation.

The modular implementation split the original `sticky_kv_logic_cummulative.py` into:

| Current file | Responsibility |
|---|---|
| `src/models/kv_cache/cache.py` | Main `STICKYKVCache_LayerWise` coordinator |
| `src/models/kv_cache/eviction_manager.py` | Window scores, decode cycle, logical ID map |
| `src/models/kv_cache/quantize_manager.py` | Q-cache quantization, promotions, rebuild |
| `src/models/kv_cache/tracking_manager.py` | Optional token ledger and global counter |
| `src/models/kv_cache/allocator.py` | CFO budget allocation |
| `src/models/kv_cache/eviction.py` | Pure prompt-stage and helper eviction functions |
| `src/models/attention/module.py` | Current SDPA `STICKYLlamaAttention` |
| `src/models/attention/ops.py` | Attention matmul/softmax helpers |
| `src/models/attention/rope.py` | RoPE wrappers and Llama 3 RoPE |
| `src/models/sticky_cache.py` | `StickyDynamicLayer` and legacy `StickyCache` |
| `src/models/sticky_llama_model.py` | HF model wrapper and generation prep override |

The newly added comparison repo:

| Repo | Why it matters |
|---|---|
| `DefensiveKV/` | Shows a modern HF-compatible KV compression approach using `kvpress`, forward hooks, explicit manual decoding, and optional key rerotation |

## Config State

The sticky config values that matter most are currently intended to match:

```python
R_RATIO = 20
P_RATIO = 50
Q_RATIO = 10
QUANTIZATION_BIT_WIDTH = 4
OMEGA = 8
SINK_TOKENS = 4
tracking_flag = 0
GENERATION_CONFIG["do_sample"] = False
GENERATION_CONFIG["max_new_tokens"] = 512
```

Two caveats:

1. `main.py` in both repos injects `LOCAL_NUM_TOKENS = 256` at runtime, but the LongBench eval runners do not use `main.py`. LongBench should therefore use `P_RATIO=50`, unless a different entrypoint is used.
2. `src/sticky_config.py` had previously drifted to `Q_RATIO=20`; it was changed back to `Q_RATIO=10`. Check `git diff` before running if a score looks inconsistent.

## Files Modified Recently

Recent local modifications include:

| File | Change |
|---|---|
| `src/models/attention/module.py` | Added `[STEP1-CACHE current]`, `[DECODE-TRACE current PRE/LOGITS/SCORES/POST/WRITE]`; earlier work added cache dispatch and StickyDynamicLayer writeback |
| `originalQevict/sticky_llama_attention.py` | Added matching `[STEP1-CACHE original]`, `[DECODE-TRACE original PRE/LOGITS/SCORES/POST]`, and continuous `_dbg_count` |
| `src/models/sticky_llama_model.py` | Added `[GEN-PREP current]` to expose `input_len`, sliced length, `cache_position`, `position_ids`, and cache type |
| `originalQevict/sticky_llama_model.py` | Added matching `[GEN-PREP original]` |
| `src/models/kv_cache/cache.py` | Has a local diff for `R_RATIO=100` decode early-exit state update |
| `src/models/kv_cache/allocator.py` | Changed `round(...)` back to `int(...)` for q-cache target parity |
| `src/sticky_config.py` | Changed `Q_RATIO` back from `20` to `10` |

Important working tree note:

- `originalQevict/` appears as untracked in this workspace.
- Do not assume all changed files are from the current turn; inspect diffs before committing or reverting anything.

## What We Have Established By Testing

### 1. Cache was initially not propagating at all

Early diagnostics showed:

```text
[DBG L0 step=0] q_len=2794 cache_type=None past_kv_shape=None
[DBG L0 step=1] q_len=1 cache_type=None past_kv_shape=None
[DBG L0 step=2] q_len=1 cache_type=None past_kv_shape=None
```

Conclusion:

- The model was decoding without past KV.
- This explained poor behavior for that stage, but not the later failures after cache propagation was fixed.

Root cause:

- Modern HF passed `past_key_values` plural.
- Our attention module accepted `past_key_value` singular.
- The plural kwarg was swallowed by `**kwargs`.

Fix:

```python
def forward(..., past_key_value=None, ..., past_key_values=None, **kwargs):
    if past_key_value is None and past_key_values is not None:
        past_key_value = past_key_values
```

Applied in:

- `src/models/attention/module.py`
- `src/models/attention/module_flash.py`

### 2. The cache object then arrived, but the read path was wrong

After the plural-kwarg fix:

```text
[DBG L0 step=0] q_len=2794 cache_type=DynamicCache past_kv_shape=None
[DBG L0 step=1] q_len=1 cache_type=DynamicCache past_kv_shape=None
```

Conclusion:

- HF was creating/passing a `DynamicCache`.
- Our code still could not read tensors from it.

Root cause:

- Kaggle/modern HF uses `DynamicCache.layers[layer_idx].keys/values`.
- Older code expected `cache.key_cache[layer_idx]` and `cache.value_cache[layer_idx]`.

Fix:

`_read_kv_from_cache` and `_write_kv_to_cache` now dispatch across:

1. New HF: `cache.layers[layer_idx].keys/values`
2. Old HF: `cache.key_cache/value_cache`
3. Legacy tuple `(k, v)`

### 3. Cache propagation then worked

After DynamicCache API handling:

```text
[DBG L0 step=0] q_len=2794 cache_type=DynamicCache past_kv_shape=None
[DBG L0 step=1] q_len=1 cache_type=DynamicCache past_kv_shape=[1,8,634,64]
[DBG L0 step=2] q_len=1 cache_type=DynamicCache past_kv_shape=[1,8,635,64]
```

Conclusion:

- Cache object was stable and readable.
- Physical KV grew between decode steps.
- Eviction fired periodically.

Representative eviction evidence:

```text
current_kv=[1,8,586,64] evicted_kv=[1,8,578,64]
current_kv=[1,8,579,64] evicted_kv=[1,8,579,64]
```

Interpretation:

- Cache grew from `578` to `586`, then evicted by `OMEGA=8`.
- This established that cache propagation and recurrent eviction were mechanically active.

### 4. No NaN/obvious tensor explosion was found

Diagnostics showed:

- No NaNs in attention output.
- Output norms were reasonable, around `0.33` in the inspected logs.
- Projection weight norms looked stable:
  - `q_proj_W ~= 7.4e1`
  - `k_proj_W ~= 4.775e1`

Conclusion:

- The failure is probably not a simple weight-loading failure or catastrophic tensor corruption.

Still not fully proven:

- Exact Q/K/V/O equality to original has not been numerically compared after load.

### 5. Position tracking was a real bug

Modern HF computes model-level `position_ids` using `past_key_values.get_seq_length()`.

With a normal `DynamicLayer`, `get_seq_length()` returns physical compressed length, e.g. `578`, while the true token position is around `2800`.

This can produce:

```text
position_ids=[578]
```

when it should be:

```text
position_ids=[2800]
```

Fix:

- Added `StickyDynamicLayer` in `src/models/sticky_cache.py`.
- It tracks:
  - `cumulative_length`: true tokens ever seen
  - `keys.shape[-2]`: physical compressed cache length
- It overrides:
  - `get_seq_length()` to return `cumulative_length`
  - `get_mask_sizes(cache_position)` to return dimensions compatible with physical KV

Additional fix:

- `get_mask_sizes` originally had the wrong signature and caused a crash in HF masking utilities.
- It now accepts a `cache_position` tensor and derives `query_length = cache_position.shape[0]`.

Conclusion:

- Wrong HF position inference was a confirmed bug.
- StickyDynamicLayer is required for the modern HF `DynamicCache` path.

### 6. Even after cache/position fixes, text quality remained poor

Observed generated text included repetitive or incoherent outputs such as:

```text
else else else else ...
```

and garbage-like code completions with repeated punctuation, URLs, digits, or fragments.

Conclusion:

- Fixing propagation alone was insufficient.
- There is likely still a semantic mismatch between current generation and original/reference behavior, or a remaining refactor bug.

## Key Numerical Clue From Earlier Logs

For the same LCC sample 2 style run:

Recent pasted bad run:

```text
Prompt Len: 1675
True sequence length: 1738
L0 step=1 K_norm ~= 8.56e2
```

Older local `debug.log` healthier/current comparison run for similar sample:

```text
Prompt Len: 1675
True sequence length: 1738
L0 step=1 K_norm ~= 1.67e3
```

Interpretation:

- Same/similar true position and similar per-token Q/K norms.
- But very different concatenated K norm.
- This suggests a different effective physical cache length or retained KV energy at decode step 1.

This is why the new step-1 physical-length diagnostic was added to both current and original paths.

## Step-1 Comparison Result

The comparable step-1 logs were added to both repos and then compared across the 20-sample LCC diagnostic.

The parsed result:

```text
debug.log         20 current STEP1 rows
debugOriginal.log 20 original STEP1 rows
same fields: True
```

Representative rows:

```text
sample 0: prompt/global_tc=2794, phys_before=634, concat_len=635
sample 1: prompt/global_tc=2129, phys_before=513, concat_len=514
sample 2: prompt/global_tc=1675, phys_before=427, concat_len=428
```

Fields that matched sample-for-sample:

| Field | Result |
|---|---|
| `pos_ids` | matched |
| `global_tc` | matched |
| `phys_before` | matched |
| `concat_len` | matched |
| `use_cache` | matched |

The only observed step-1 difference:

```text
current:  cache_type=DynamicCache
original: cache_type=tuple
```

Conclusion:

- OriginalQEvict is **not** scoring higher because it has a larger physical cache at the first generated token.
- The initial true position and compressed physical cache length are aligned.
- The earlier hypothesis "original may be using near-full prompt cache at step 1" is ruled out for this run.
- The score gap must emerge either after step 1 or from tensor-content/value differences not visible through length/position logging.

Important limitation:

- `debugOriginal.log` did not contain original per-sample generated text or first generated token IDs.
- `debug.log` had current per-sample generations and scores, but original only had aggregate score plus step-1 state.
- Therefore the exact first token or first step where text diverges is still unknown.

## Current Decode Trace Diagnostics

### Current modular path

File:

- `src/models/attention/module.py`

Existing step-1 line:

```text
[STEP1-CACHE current] cache_type=... q_len=... pos_ids=... global_tc=... phys_before=... concat_len=... use_cache=...
```

New decode trace lines:

```text
[DECODE-TRACE current PRE step=...]
[DECODE-TRACE current LOGITS step=...]
[DECODE-TRACE current SCORES step=...]
[DECODE-TRACE current POST step=...]
[DECODE-TRACE current WRITE step=...]
```

The current trace prints for:

- The first 10 layer-0 single-token decode steps.
- Any step where `tokens_since_last_review` is at `OMEGA - 1` or `OMEGA`, so the review boundary is visible.

Fields:

| Field | Meaning |
|---|---|
| `cache_type` | HF cache object type passed to attention |
| `q_len` | Decode query length; should be `1` |
| `pos_ids` | RoPE position ID used by our attention |
| `global_tc` | Internal true token counter before cache update for this step |
| `phys_before` | physical KV length before appending current token |
| `concat_len` | physical KV length after appending current token |
| `tokens_since` | decode tokens accumulated toward the next `OMEGA` review |
| `gen_step` | sticky eviction manager decode-step counter |
| `q_windows` | number of q-cache windows currently present |
| `qcache_active` | whether q-cache tensors exist and joint softmax should run |
| `cache_position` | `cache_position` received by attention through `kwargs` |
| `main_shape` | raw main attention logits shape before masking/softmax |
| `logits_norm`, `max`, `min` | raw main attention logits diagnostics |
| `scores_shape` | shape of `scores_for_cache` passed into Sticky KV |
| `q_scores_shape` | shape of q-cache attention scores, if q-cache is active |
| `output_attn_shape` | shape of attention weights used for output/diagnostics |
| `evicted_len` | physical length returned by Sticky KV after possible eviction |
| `cache_seq_len` | current-only DynamicCache reported sequence length after writeback |
| `cache_seen` | current-only `_seen_tokens` value where available |
| `use_cache` | whether attention is writing cache |

### Original path

File:

- `originalQevict/sticky_llama_attention.py`

Existing step-1 line:

```text
[STEP1-CACHE original] cache_type=... q_len=... pos_ids=... global_tc=... phys_before=... concat_len=... use_cache=...
```

New decode trace lines:

```text
[DECODE-TRACE original PRE step=...]
[DECODE-TRACE original LOGITS step=...]
[DECODE-TRACE original SCORES step=...]
[DECODE-TRACE original POST step=...]
```

There is no original `WRITE` line because original returns the tuple cache directly through the old HF path. That absence is intentional and itself marks one of the key lifecycle differences.

### Generation Prep Trace

Files:

- `src/models/sticky_llama_model.py`
- `originalQevict/sticky_llama_model.py`

Lines:

```text
[GEN-PREP current step=...]
[GEN-PREP original step=...]
```

Fields:

| Field | Meaning |
|---|---|
| `input_len` | full generated sequence length visible to `prepare_inputs_for_generation` |
| `sliced_len` | length actually passed to the model after override; should be `1` during decode |
| `global_tc` | current-only true counter read from layer 0 |
| `cache_position` | model-level cache position after HF/specific overrides |
| `position_ids` | model-level position IDs after HF/specific overrides |
| `pkv_type` | type of `past_key_values` passed by generation |

Why this matters:

- Current explicitly sets `cache_position=[global_tc]`.
- Original does not explicitly set `cache_position`.
- Both attention modules override `position_ids` internally from `global_token_counter`, but HF may still use `cache_position` for slicing/masking/cache bookkeeping before attention runs.

## Runtime Fixes Already Applied

### Conditional mapping allocation

Problem:

- `mapping` was allocated/populated even with `tracking_flag=0`.
- Original only created mapping when tracking was enabled.

Fix:

- `mapping = ... if self._tracking_flag else None`
- All mapping writes guarded.

Status:

- Fixed runtime issues.
- Not proven to affect score when tracking is disabled.

### Promoted Q-cache mapping write

Problem:

- Mapping write for promoted Q-cache windows was suppressed during a previous rerotation experiment.
- Rerotation was later removed.

Fix:

- Restored mapping write for promoted Q-cache windows.

Status:

- Correct for parity/tracking.
- Probably not score-critical when `tracking_flag=0`.

### Path C bounds check

Problem:

- In Q-cache rebuild Path C, `v_fp = past_key_values[1][0, h_val, ps:ps + omega]` could return an empty slice when out of bounds.
- This caused shape mismatch with archived scale/zp tensors.

Fix:

- Guarded with `if ps + omega <= seq_len`; otherwise zero-fill.

Status:

- Fixed crash.
- Zero-fill fallback may hide a logic issue if it happens often; should be counted if quality remains bad.

### Physical cache rebuild OOB guard

Problem:

- `first_phys + omega` could exceed `seq_len`.
- Vectorized gather could read out of bounds.

Fix:

- Added `in_bounds = (first_phys + omega) <= seq_len`.
- Applied to SDPA and FA2 cache paths and quantize Path B.

Status:

- Fixed crash.
- Frequency of out-of-bounds fallback should be measured if score remains low.

### `valid_old_windows` `.max()` vs `.min()`

Problem:

- Refactor used `.max()` where original used `.min()`.
- `.max()` could read NaN-padded slots for heads with fewer valid windows.

Fix:

- Restored `.min()`.

Status:

- Did not noticeably improve score in the observed run.
- Still considered correct parity fix.

### `R_RATIO=100` early-exit window state

Problem:

- In `src/models/kv_cache/cache.py`, there is a local diff updating `window_scores` before returning in `total_cache_ratio == 100` decode early-exit.

Status:

- This is a local diff and should be reviewed before treating it as part of the main fix.
- It is not expected to affect `R_RATIO=20` runs.

## Logic Parity Audits Completed

The following areas were inspected and appear intended to match the original:

| Area | Status |
|---|---|
| CFO allocator | Same sequential budget formula; current has been changed back to `int()` parity |
| Prefill window scoring | Slice/sum/window population matches original |
| Local history seeding | Matches original conceptually |
| Decode scoreboard | Matches original reconstruction and challenger competition |
| Top-k/top-q selection | Matches original; includes dedup guard |
| Physical cache rebuild | Structure matches original, with extra bounds guards |
| Q-cache Path A/B/C | Structure matches original, with extra bounds guards |
| Main attention logits | GQA reshape/matmul matches original |
| Standard softmax path | Matches original |
| Q-cache joint softmax path | Matches original structure |
| Llama 3 RoPE formula | Audited as matching |
| `apply_rotary_pos_emb_single` | Re-exported from shared helper |
| Causal mask helper | Re-exported from shared helper |

Important caution:

- These are code-structure parity checks, not full numerical equivalence tests.
- A single tensor indexing or dtype/layout difference can still survive these audits.

## DefensiveKV Findings

The added `DefensiveKV/` repo is useful because it shows modern HF can support KV compression, but its integration model differs sharply from current Sticky.

### DefensiveKV compresses through hooks

File:

- `DefensiveKV/kvpress/presses/base_press.py`

Behavior:

- Registers a forward hook on each attention layer.
- Lets the normal HF attention forward run.
- After the layer forward, directly mutates:

```python
cache.layers[module.layer_idx].keys = keys
cache.layers[module.layer_idx].values = values
```

Meaning:

- DefensiveKV does not replace the attention module.
- It compresses cached tensors after normal HF cache update.

### DefensiveKV does manual decoding, not `model.generate()`

File:

- `DefensiveKV/kvpress/pipeline.py`

Behavior:

- Prefills context into a cache.
- Compresses during that prefill.
- Then manually feeds question/generation tokens with explicit `position_ids`.

Representative logic:

```python
position_ids = torch.arange(
    context_length, context_length + question_ids.shape[1], device=self.model.device
).unsqueeze(0)
```

Then decode loop:

```python
outputs = self.model(
    input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
    past_key_values=cache,
    position_ids=position_ids + i,
)
```

Meaning:

- DefensiveKV bypasses much of HF `generate()` cache-position inference.
- It controls true positions explicitly.

### DefensiveKV separates context and question

File:

- `DefensiveKV/evaluation/longbench/create_huggingface_dataset.py`

For LCC:

- Context prompt:

```text
Please complete the code given below.
{context}
```

- Question prompt:

```text
Next line of code:
```

Meaning:

- DefensiveKV typically compresses context first, then feeds the question/suffix uncompressed.
- Current Sticky `src/data/data_loader.py` builds one combined prompt:

```text
Please complete the code given below.
{context}Next line of code:
```

and compresses it all as one prefill.

This difference can matter a lot for code completion because the suffix/instruction immediately before generation should often remain local and uncompressed.

### DefensiveKV has explicit rerotation support

File:

- `DefensiveKV/kvpress/presses/key_rerotation_press.py`

It documents a valid alternative:

1. Unrotate keys from original RoPE positions.
2. Prune.
3. Rerotate kept keys into compact positions.
4. Decode using compressed positions.

Current Sticky is **not** using rerotation. It stores retained keys with their original absolute RoPE positions and decodes with true absolute `position_ids`.

Both approaches can be valid, but they must not be mixed.

## Important Non-Conclusions

Do not claim:

- "HF cache contract is definitely the only difference."
- "Eviction/scoring logic is proven numerically identical."
- "OriginalQevict used near-full cache at step 1."
- "DefensiveKV proves Sticky's current generation path must work."

What we can claim:

- Cache propagation was broken and then fixed.
- Modern DynamicCache API handling was broken and then fixed.
- Position tracking through compressed cache was broken and then fixed with `StickyDynamicLayer`.
- Current output can still be incoherent.
- Step-1 physical cache length and true position now match between original and current for the inspected 20 LCC samples.
- The remaining score gap must be found after step 1 or in tensor contents/logits/scores that the earlier step-1 length/position log did not expose.

## Open Hypotheses

| ID | Hypothesis | Why plausible | How to test |
|---|---|---|---|
| H1 | Original and current have different step-1 physical cache length | Earlier K-norm clue suggested this | **Ruled out for the inspected run**: `[STEP1-CACHE]` matched across 20 samples |
| H2 | Current recurrent decode eviction corrupts useful context | Cache works but quality worsens | Disable decode-time eviction after prefill |
| H3 | Prompt/question compression schedule hurts LCC | DefensiveKV keeps question/suffix uncompressed | Implement DefensiveKV-style manual context/question split |
| H4 | Q-cache promotion/rebuild has a subtle bug | Garbage appears after cache is active; bounds fallbacks exist | Run with `Q_RATIO=0`; count Path B/C fallback events |
| H5 | RoPE/cache-position semantics still mismatched after step 1 | Compression with RoPE is fragile; current explicitly sets `cache_position` and original does not | Compare `[GEN-PREP]` and `[DECODE-TRACE ... PRE]` for steps 1-10 |
| H6 | Weight load is subtly wrong | Custom attention modules are replaced after parent init | Numerically compare q/k/v/o weights to HF/original |
| H7 | `generate()` introduces hidden behavior DefensiveKV avoids | DefensiveKV uses manual greedy decoding | Add manual greedy Sticky eval |
| H8 | Bounds guards prevent crashes but zero-fill important windows | Zero-filled windows can poison attention | Instrument fallback counts and window IDs |
| H9 | Current and original logits/scores diverge before QEvict consumes scores | Same lengths/positions do not prove same tensor content | Compare `[DECODE-TRACE ... LOGITS/SCORES]` |
| H10 | Current writes correct `evicted_kv` but DynamicCache reports/uses different sequence state afterward | Current mutates cache in place; original returns tuple cache | Compare `[DECODE-TRACE current WRITE]` with next-step original/current `PRE` |

## Immediate Next Step

Run the same small LCC diagnostic in both implementations after the new trace hooks:

1. Current modular:

```powershell
python -m src.eval.run_longbench_sticky
```

or the Kaggle equivalent entrypoint.

2. Original:

```powershell
cd originalQevict
python Results/run_longbench_sticky.py
```

Then compare the first 10 decode steps and the first `OMEGA=8` review boundary.

The most important fields are:

```text
GEN-PREP:      input_len, sliced_len, cache_position, position_ids, pkv_type
PRE:           pos_ids, global_tc, phys_before, concat_len, tokens_since, gen_step, q_windows, qcache_active
LOGITS:        main_shape, logits_norm, max, min
SCORES:        scores_shape, q_scores_shape, output_attn_shape
POST:          evicted_len, tokens_since, gen_step, q_windows
current WRITE: cache_seq_len, cache_seen, written_global_tc
```

Interpretation priority:

1. If `[GEN-PREP]` differs on `cache_position` in a way that correlates with later divergence, test matching original's generation prep more closely.
2. If `[DECODE-TRACE ... PRE]` differs after step 1 while step 1 matched, focus on cache lifecycle/writeback.
3. If `PRE` matches but `LOGITS` diverges, compare K/V tensor contents and RoPE inputs.
4. If `LOGITS` matches but `SCORES` or `POST` diverges, inspect softmax/q-cache/QEvict inputs.
5. If everything matches through `POST` but current `WRITE` causes next-step divergence, focus on `DynamicCache` mutation/get-seq-length/seen-token semantics.

If these still match, the next diagnostic should compare tensor content directly:

- Selected survivor window IDs after prefill.
- `window_scores` top-k IDs/values for layer 0.
- Q-cache loser IDs/values.
- First decode attention top-k physical indices.
- Rebuild fallback counts.
- LM-head top-k logits and selected token IDs for both current and original.

## Suggested Follow-Up Experiments

### Experiment A: Disable decode eviction after prefill

Goal:

- Determine whether prefill compression alone is okay and recurrent rebuild is the culprit.

Implementation idea:

- In `STICKYKVCache_LayerWise.__call__`, after prefill, return `past_key_values` directly during decode while still appending new tokens.
- Or set a flag to skip the `tokens_since_last_review == omega` rebuild path.

Expected outcomes:

| Result | Interpretation |
|---|---|
| Score improves strongly | Recurrent decode eviction/rebuild/Q-cache likely culprit |
| Score remains bad | Prefill compression, generation protocol, or positions likely culprit |

### Experiment B: `Q_RATIO=0`

Goal:

- Remove Q-cache quantization/promotion from the equation.

Expected outcomes:

| Result | Interpretation |
|---|---|
| Score improves | Q-cache quantization/rebuild/promotion bug likely |
| Score remains bad | BF16 survivor/local/cache protocol likely |

### Experiment C: Manual greedy decoding

Goal:

- Match DefensiveKV's generation style and avoid `model.generate()`.

Approach:

1. Tokenize prefill prompt.
2. Run model forward with cache.
3. Feed one token at a time with explicit `position_ids`.
4. Greedy argmax.

Expected outcomes:

| Result | Interpretation |
|---|---|
| Manual decode improves | HF `generate()` integration still differs |
| Manual decode remains bad | Compression/rebuild math more likely |

### Experiment D: Context/question split for LCC

Goal:

- Test whether compressing `Next line of code:` inside prefill hurts code completion.

Approach:

- Prefill only `context_prompt`.
- Then feed `question_prompt` with explicit positions after compression.
- Then decode.

Expected outcomes:

| Result | Interpretation |
|---|---|
| Score improves | Prompt schedule mismatch vs DefensiveKV/original matters |
| No improvement | Look at cache content and rebuild |

## How To Interpret The Next Logs

Step-1 length/position match is already established. The next logs should be interpreted as a divergence search.

Example healthy beginning:

```text
[GEN-PREP original step=0] input_len=2795 sliced_len=1 cache_position=... position_ids=[2794] pkv_type=tuple
[GEN-PREP current step=0]  input_len=2795 sliced_len=1 cache_position=[2794] position_ids=[2794] pkv_type=DynamicCache

[DECODE-TRACE original PRE step=1] ... pos_ids=[2794] global_tc=2794 phys_before=634 concat_len=635 ...
[DECODE-TRACE current PRE step=1]  ... pos_ids=[2794] global_tc=2794 phys_before=634 concat_len=635 ...
```

This would mean:

- Generation prep and attention-level step-1 state are aligned except for the expected cache object type.
- Continue comparing `LOGITS`, `SCORES`, `POST`, and next-step `PRE`.

Example cache-position mismatch:

```text
[GEN-PREP original step=0] ... cache_position=[634] position_ids=[2794] ...
[GEN-PREP current step=0]  ... cache_position=[2794] position_ids=[2794] ...
```

This would mean:

- Attention may still receive the same `pos_ids`, but HF internals may see different `cache_position`.
- Test whether matching original's `prepare_inputs_for_generation` behavior changes score.

Example post-write mismatch:

```text
[DECODE-TRACE current POST step=8] ... evicted_len=635 tokens_since=0 gen_step=8 ...
[DECODE-TRACE current WRITE step=8] cache_seq_len=2802 cache_seen=2802 written_global_tc=2802
[DECODE-TRACE current PRE step=9] ... phys_before=643 concat_len=644 ...
```

This would mean:

- QEvict returned one physical length, but the next step read a different physical length.
- Focus on `_write_kv_to_cache`, `StickyDynamicLayer`, and DynamicCache layer state.

Example logits mismatch with matching lengths:

```text
[DECODE-TRACE original LOGITS step=2] main_shape=(1, 32, 1, 636) logits_norm=...
[DECODE-TRACE current LOGITS step=2]  main_shape=(1, 32, 1, 636) logits_norm=very different
```

This would mean:

- Cache length and position are not enough; K/V tensor contents or RoPE application differ.
- Compare retained K/V tensors, survivor IDs, and window-score selections.

## Final Current Position

We should be careful and empirical:

- The HF cache path was definitely broken in multiple ways and has been partially repaired.
- The current bad quality is not yet explained.
- DefensiveKV suggests a safer reference protocol: compress during prefill, manually decode with explicit positions, and avoid compressing the question/suffix when evaluating LongBench-style tasks.
- The step-1 logs have ruled out initial physical cache length/position mismatch.
- The newest decode-trace logs are the next decisive measurement.
