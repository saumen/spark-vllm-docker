# CUDAGraphMode Incompatibility with FlashInferBackend + Spec-Decode

## Symptom

Startup warnings when using `--compilation-config '{"cudagraph_mode":"FULL"}'`
with vLLM 0.25.x on DGX Spark:

```
(EngineCore pid=118) WARNING 08-01 08:10:34 [compilation.py:1362]
CUDAGraphMode.FULL is not supported with FlashInferBackend backend
(support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
setting cudagraph_mode=FULL_AND_PIECEWISE

(EngineCore pid=118) WARNING 08-01 08:10:34 [compilation.py:1409]
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
attention backend FlashInferBackend (support:
AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
setting cudagraph_mode=PIECEWISE
```

## Root Cause

vLLM's compilation engine selects CUDAGraphMode based on the attention backend
and whether speculative decoding is active:

1. **`FULL`**: Requires `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`.
   FlashInferBackend does not support this — it only provides
   `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` for single-token
   decode, not the full uniform decode graph needed by `FULL`.

2. **`FULL_AND_PIECEWISE`**: A hybrid mode that still requires
   `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` support. Additionally,
   when spec-decode is active (DFlash or MTP), this mode is blocked entirely
   — spec-decode uses variable numbers of tokens per step, which `FULL`
   and `FULL_AND_PIECEWISE` cannot handle.

3. **`PIECEWISE`**: The only mode compatible with both FlashInferBackend
   and spec-decode. It compiles graphs per-sequence-group rather than
   uniformly, which handles variable token counts.

## Solution

Hardcode `PIECEWISE` explicitly in the launch script:

```bash
CUDAGRAPH_MODE='{"cudagraph_mode":"PIECEWISE"}'

docker run ... \
  --compilation-config "$CUDAGRAPH_MODE" \
  --speculative-config '{"model":"DRAFT","num_speculative_tokens":3,"method":"dflash"}' \
  ...
```

## Why This Matters on DGX Spark

Both Laguna S 2.1 and Unsloth Qwen3.6 NVFP4 launches use:
- FlashInferBackend (default attention backend)
- Speculative decoding (DFlash for Laguna, MTP for Unsloth)

Therefore `PIECEWISE` is always the correct cudagraph mode. Setting `FULL`
triggers a runtime fallback that wastes 1-2 minutes on JIT recompilation
during startup.

## Verification

After changing the compilation config:

```bash
# No CUDAGraphMode warnings in logs
docker logs vllm-laguna-s21 2>&1 | grep "CUDAGraphMode" || echo "No warnings"
```
