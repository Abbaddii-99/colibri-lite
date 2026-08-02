# Colibri-Lite — Modular GLM Inference Engine

Drop-in modular refactoring of the [Colibri](https://github.com/anomalyco/colibri) C inference engine. Extracted 12 independent modules from the monolithic `glm.c` while preserving **byte-identical output** and the full serve protocol.

## Build

Requires GCC 13+ and make.

```bash
cd c
make                  # CPU-only build
make cuda             # CUDA build (requires nvcc)
```

Output: `c/glm` (or `glm.exe` on Windows).

### CUDA

```bash
make cuda CUDA_PATH="/usr/local/cuda"
```

CUDA offloads dense matmuls and expert weights to GPU. Set `CUDA_DEVICES="0,1"` for multi-GPU.

## Quick Start

```bash
# Set the model directory
export SNAP=/path/to/glm-model

# Interactive chat
./glm

# Server mode (for openai_server.py)
SERVE=1 ./glm

# Single prompt
PROMPT="Hello, world!" NGEN=128 ./glm
```

The server protocol is compatible with the existing `openai_server.py` from `colibri-main/c/`:

```bash
python openai_server.py --model /path/to/glm-model --engine c/glm
```

## Architecture

```
c/
├── include/           # Headers
│   ├── model.h        # Core types: Model, Layer, QT, Cfg, KVState
│   ├── load.h         # Model loading
│   ├── cache.h        # Expert cache
│   ├── kv_cache.h     # KV cache
│   ├── attention.h    # Attention (MLA absorption)
│   ├── moe.h          # Mixture of Experts
│   ├── quant.h        # Quantization (int8/int4/int2)
│   ├── tensor.h       # Tensor ops (matmul, RMSNorm)
│   ├── sampling.h     # Top-K, Top-P, argmax
│   ├── spec.h         # Speculative decoding (draft-verify)
│   ├── pipeline.h     # I/O pipeline, pilot prefetch
│   ├── engine.h       # Main entry point, chat, serve, score
│   ├── st.h           # JSON + safetensors reader
│   ├── tier.h         # Expert tier management
│   ├── compat.h       # OS portability shims
│   └── backend_cuda.h # CUDA API
├── src/               # Implementations (12 .c files)
├── tests/
│   ├── smoke_test.py  # Functional tests (no model needed)
│   └── regression_test.py  # Output-identity validation (needs model)
└── glm_client.py      # Python chat client
```

### Module overview

| Module | File | Purpose |
|--------|------|---------|
| Engine | `engine.c` | `main()`, modes (serve/chat/score/text), usage tracking |
| Load | `load.c` | Model init: config.json, safetensors, quantized weights |
| KV Cache | `kv_cache.c` | KV state alloc/bind, forward pass, disk persistence |
| Attention | `attention.c` | MLA attention (with KV-absorption), DSA indexer |
| MoE | `moe.c` | Router, expert dispatch, shared experts |
| Cache | `cache.c` | Expert LRU: load/prefetch/evict, slot management |
| Quant | `quant.c` | Quantized matmul (int8 dot-product, int4 lookup) |
| Tensor | `tensor.c` | `matmul_qt`, `rmsnorm`, `embed_row` |
| Sampling | `sampling.c` | `sample_topk`, `argmax` |
| Spec | `spec.c` | Speculative decoding (draft-verify loop) |
| Pipeline | `pipeline.c` | I/O overlap, pilot prefetch routing |
| Globals | `globals.c` | Global tunables, RNG, timer |

## Serve Protocol

The engine accepts commands on stdin:

| Command | Format | Description |
|---------|--------|-------------|
| Plain text | `text...\n` | Chat mode (auto-formats with template) |
| PROMPT | `\x02PROMPT {len} {max_tokens} {temp} {top_p} [slot]\n{raw_bytes}\n` | Raw prompt (byte-exact), per-request params |
| MORE | `\x02MORE\n` | Continue truncated response |
| RESET | `\x02RESET\n` or `\x01\x01RESET\n` | Clear context |

Each response ends with `\x01\x01END\x01\x01\n` followed by a `STAT` line:
```
STAT {tokens} {tok/s} {cache_hit%} {rss_gb} [{prompt_tokens} {did_truncate}]
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SNAP` | — | Model directory (required) |
| `TEMP` | auto | Temperature (0–2) |
| `NUCLEUS` | 0.9 | Top-P threshold |
| `TOPK` | 0 | Top-K sampling (0 = disabled) |
| `TOPP` | 0 | Top-P sampling (0 = disabled) |
| `NGEN` | 512 | Max generation tokens |
| `DRAFT` | auto | Speculative draft tokens (auto: 3 with MTP, 0 without) |
| `SPEC` | 1 | Speculative decoding on/off |
| `SEED` | random | RNG seed |
| `SERVE` | — | Enable server mode |
| `CTX` | 4096 | Max context length |
| `KV_SLOTS` | 1 | Multi-slot KV cache (1–16) |
| `KVSAVE` | 1 | Persist KV cache to disk |
| `THINK` | 0 | Emit `<think>` tags |
| `PILOT` | 0 | Pilot prefetch (expert cache hint) |
| `PIPE` | 0 | I/O pipeline (overlap load + matmul) |
| `CUDA_DEVICES` | 0 | Comma-separated GPU IDs (CUDA build) |
| `PIN` | — | Expert pin file path |
| `REPIN` | 0 | Live re-pin interval (tokens) |
| `ABSORB` | auto | MLA KV-absorption (auto: on for S≤4) |

## Tests

```bash
# Smoke tests (no model required)
python c/tests/smoke_test.py

# Regression test (requires model snapshot)
SNAP=/path/to/model python c/tests/regression_test.py
```
