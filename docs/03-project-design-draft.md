# New Project Design Draft — Colibri-Lite

**Vision:** Run large MoE language models exclusively on consumer hardware (16-32 GB RAM, single GPU, NVMe SSD)

---

## 1. Core Goals

| # | Goal | Metric |
|---|------|--------|
| 1 | Cold decode ≥ 10 tok/s | Storage-bound throughput |
| 2 | Warm decode ≥ 25 tok/s | PCIe-bound throughput |
| 3 | Support any HuggingFace MoE | Model-agnostic converter |
| 4 | RAM ≤ 16 GB | Consumer 16 GB floor |
| 5 | Zero Python deps at runtime | Self-contained binary |
| 6 | Single GPU ≤ 24 GB VRAM | RTX 4090-class |

---

## 2. Scope Boundaries

| In Scope | Out of Scope |
|----------|-------------|
| MoE inference engine | Training or fine-tuning |
| HuggingFace model conversion | Distributed inference (multiple machines) |
| INT8 / INT4 quantization | FP16 training precision |
| Speculative decoding (MTP) | KV cache offloading to disk mid-generation |
| OpenAI-compatible HTTP API | LangChain / LLM framework integrations |
| Linux + macOS + Windows | Mobile / embedded / WebAssembly |

---

## 3. Architecture — 4-Tier Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Tier 1: User Interface                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  CLI     │  │  Chat    │  │  Serve   │  │  Convert    │  │
│  │  (chat)  │  │  (stdin) │  │  (HTTP)  │  │  (model)    │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Tier 2: Inference Scheduler                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  Router  │  │  Spec    │  │  Cache   │  │  Pipeline   │  │
│  │  (MoE)   │  │  (MTP)   │  │  (Tier)  │  │  (PIPE=1)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Tier 3: Compute Engine                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  Attn    │  │  MoE FFN │  │  KV      │  │  Backend    │  │
│  │  (MLA)   │  │  (INT8)  │  │  Cache   │  │  (CUDA/CPU) │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Tier 4: I/O & Storage                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │  Reader  │  │  Format  │  │  Token   │  │  Grammar    │  │
│  │  (shard) │  │  (conv)  │  │  (BPE)   │  │  (GBNF)     │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Component Descriptions

### Tier 1 — User Interface

| Component | Responsibility |
|-----------|---------------|
| **CLI** | Argument parsing, dispatch to chat/serve/convert |
| **Chat** | Interactive REPL with history, system prompt, multi-turn |
| **Serve** | OpenAI-compatible HTTP API (stdlib `http.server`) |
| **Convert** | HuggingFace → Colibri format converter (Python script) |

### Tier 2 — Inference Scheduler

| Component | Responsibility |
|-----------|---------------|
| **Router** | MoE top-k routing with batch-union optimization |
| **Spec** | Multi-token prediction (MTP) speculative decoding |
| **Cache** | Tiered expert cache with heat decay and LRU eviction |
| **Pipeline** | Asynchronous I/O pipeline (PIPE=1) with eventfd synchronization |

### Tier 3 — Compute Engine

| Component | Responsibility |
|-----------|---------------|
| **Attention** | Multi-head Latent Attention (MLA) with KV cache absorption |
| **MoE FFN** | INT8 quantized feed-forward network with IDOT kernels |
| **KV Cache** | KV cache storage, FP8 compression, disk save/load |
| **Backend** | CPU (SIMD) and optional CUDA (quantized matmul) |

### Tier 4 — I/O & Storage

| Component | Responsibility |
|-----------|---------------|
| **Reader** | Hash-indexed sharded safetensors reader (pread-based) |
| **Format** | Colibri model format specification + converter |
| **Token** | Byte-level BPE tokenizer (cl100k base, FNV-1a hashing) |
| **Grammar** | GBNF grammar parser and PDA byte-level constraint walker |

---

## 5. Decode-Step Data Flow

```
Tokenize (≈1ms)
    │
    ▼
Attention: Q·K^T → Score·V (≈20ms)
    │
    ▼
Router: top-k expert selection (≈3ms)
    │
    ▼
Load expert weights (≈60ms)  ←─── PREFETCH next experts
    │                              ↑ PIPE overlap
    ▼
Expert FFN1: matmul (≈20ms)
    │
    ▼
Expert FFN2: matmul (≈20ms)
    │
    ▼
Sample token (≈1ms)
    │
    ▼
KV cache update (≈5ms)
    │
    ▼
[return to Tokenize]
```

**Warm decode total:** ~130ms per token → ~7.7 tok/s  
**With PIPE overlap:** ~100ms → ~10 tok/s  
**Hot decode (cache hit):** ~70ms → ~14 tok/s

---

## 6. Roadmap

### Phase 0: Foundation (Weeks 1-2)

- Repository structure and build system
- Clone Colibri core components as starting point
- Unit tests for each component
- CI pipeline (basic compilation + smoke test)

**Exit criteria:** `make && ./glm chat` works with GLM-5.2

### Phase 1: Refactoring (Weeks 3-4)

- Split `glm.c` into modular components
- Extract: `attention.c`, `moe.c`, `cache.c`, `quant.c`, `io.c`, `sampling.c`, `spec.c`, `model.c`
- No behavior changes — identical output at every step
- Add internal documentation

**Exit criteria:** Same output as monolithic version, all tests pass

### Phase 2: Improvements (Weeks 5-8)

- PIPE=1 enabled by default
- Eventfd-based PILOT thread (replace usleep polling)
- DROP=1 enabled by default
- Arena allocator for hot path
- Disable n-gram speculation
- OMP_PROC_BIND=spread default
- Expert prefetch

**Exit criteria:** 2× cold decode improvement, stable memory usage

### Phase 3: Model Converter (Weeks 9-10)

- Python converter script: HuggingFace → Colibri format
- Support: DeepSeekMoE, QwenMoE, Mixtral, DBRX
- Generic ONNX export path for arbitrary MoE models

**Exit criteria:** Can load and run 3 different MoE models end-to-end

### Phase 4: Polish & Package (Weeks 11-12)

- KV cache FP8 compression
- macOS ARM (Apple Silicon) support with NEON IDOT kernels
- Windows support (MSVC compatibility)
- `pip install colibri-lite` or Homebrew formula
- Documentation and usage examples

**Exit criteria:** MVP release with 3 supported models, packaged installable

---

## 7. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GLM-5.2-specific code hard to generalize | Medium | High | Abstract model access behind generic interface from start |
| Converter complexity underestimated | High | Medium | Start with one architecture, expand iteratively |
| Page cache behavior differs on Windows/macOS | Medium | Medium | Abstract cache control behind `compat.h`; test each platform |
| CUDA backend effort high | Medium | Medium | CPU-only is acceptable MVP; CUDA is Phase 4 |
| INT8 accuracy loss on non-GLM models | Low | Medium | Validation harness; fall back to FP16 if needed |
| Community interest low | Medium | High | Focus on quality; release as developer tool, not product |
| Maintenance burden of multiple model formats | Low | Medium | Single ONNX-based format; convert only |
| NUMA effects on consumer dual-CCX (AMD) | Medium | Low | Document optimal thread placement |
| Legal concerns with model redistribution | Low | Medium | Converter only; no model weights included |

---

## 8. MVP Definition

### Must Have
- [ ] Load and run GLM-5.2 (INT8) on RTX 4090 + 32 GB RAM
- [ ] Cold decode ≥ 10 tok/s
- [ ] Warm decode ≥ 20 tok/s
- [ ] Memory stable (< 16 GB RSS over 1-hour session)
- [ ] Interactive chat mode
- [ ] OpenAI-compatible serve mode
- [ ] Single binary deployment

### Nice to Have
- [ ] Model converter for DeepSeekMoE
- [ ] CUDA backend (CPU-only acceptable)
- [ ] macOS support
- [ ] Grammar-constrained generation

### Out of Scope
- Multi-GPU inference
- Training or fine-tuning
- Non-MoE models (dense transformers)
- Quantization-aware training

### Exit Criteria
```
make && ./glm chat --model gl5-52b-int8
> Hello, who are you?
I am GLM-5.2, a multilingual language model...
> [CTRL+D]
Memory: stable at 12.4 GB RSS
Throughput: 12.5 tok/s (cold), 22.3 tok/s (warm)
GPU: 14.8 GB VRAM used, 68% utilization
```
