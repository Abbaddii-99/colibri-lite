# Strategy Plan — Colibri-Lite Execution

**Date:** 2026-07-13  
**Phase:** PLANNING → EXECUTION  

---

## Core Strategy

**Preserve the 95% value of Colibri's engine while making it multi-model, modular, and maintainable.**

Do NOT rewrite from scratch. Do NOT keep the monolithic 2817-line `glm.c`. Instead: refactor, generalize, improve.

---

## Execution Phases

### Phase A: Structural Refactoring (Weeks 1-2)

Goal: Split `glm.c` into clean components without changing a single output bit.

#### Module Map

```
glm.c  →  engine.c           (main, dispatch, serve loop)
         + attention.c/.h    (MLA QKV, score·V, KV cache)
         + moe.c/.h          (router, batch-union, expert dispatch)
         + cache.c/.h        (LRU, heat decay, PIN/REPIN from tier.h)
         + quant.c/.h        (INT8 matmul, IDOT, dequant)
         + io.c/.h           (safetensors reader from st.h)
         + sampling.c/.h     (temperature, top-p, top-k, greedy)
         + spec.c/.h         (MTP, grammar verification)
         + model.c/.h        (model_init, layer_forward, config structs)
         + kv_cache.c/.h     (KV alloc, FP8 compression, disk save/load)
         + backend_cuda.{h,cu} (as-is)
         + tok.h             (as-is, BPE tokenizer)
         + grammar.h         (as-is, GBNF parser)
         + compat.h          (as-is, platform shims)
         + json.h            (as-is, JSON parser)
```

#### Refactoring Rules

1. **No behavior changes** — identical output for identical input
2. **Each module gets a .h (public API) and .c (implementation)**
3. **All globals go into a `ModelState` struct** passed by pointer
4. **Build: single `make` command** — no autotools, no cmake
5. **Tests: compare binary output** before and after each split

#### Acceptance
```
Before: ./glm "Hello" → "Hello! How can I help?"
After:  ./glm "Hello" → "Hello! How can I help?"  ✓
```

---

### Phase B: Quick Wins — Configuration Changes (Week 3)

These require zero or minimal code changes:

| Change | File | Gain | Risk |
|--------|------|-----:|------|
| `PIPE=1` default | `Makefile` | ~1.5-2× cold | Low (tested but off) |
| `DROP=1` default | `Makefile` | Stability | Low (prevents OOM) |
| Disable n-gram draft | `spec.c` | ~2-5% | Low (5% acceptance) |
| `OMP_PROC_BIND=spread` | `Makefile` | ~10-20% | Low (NUMA) |
| `THREADS=4` default | `Makefile` | Baseline | Low |

---

### Phase C: Infrastructure Improvements (Weeks 4-6)

| Item | Effort | Gain | Description |
|------|--------|-----:|-------------|
| Eventfd PILOT thread | 2 days | ~10-15% | Replace `usleep(200)` with `eventfd` semaphore |
| Arena allocator | 3 days | ~5-10% | Pre-allocated buffers for matmul intermediates |
| Expert prefetch | 2 days | ~20% | Start loading next step's experts during attention |
| Parallel shard reader | 3 days | ~20% | Load multiple shards concurrently |
| madvise DONTNEED after DROP | 1 day | Stability | Explicit page release |

---

### Phase D: Model Converter (Weeks 7-8)

#### Converter Architecture

```
huggingface_model/          colibri_model.cfl
├── config.json       →     ├── model.cfg
├── model.safetensors →     ├── weights.bin  (INT8 quantized)
├── tokenizer.json    →     ├── tokenizer.bin
└── *.safetensors     →     └── shards/      (experts)
```

#### Supported Models (in order)

1. **DeepSeekMoE** (DeepSeek-V2, V3) — closest to GLM architecture
2. **QwenMoE** (Qwen2.5-MoE) — popular, well-documented
3. **Mixtral 8×7B** — most common MoE reference
4. **DBRX** — large MoE with fine-grained experts

#### Converter Implementation

- Python script (only Python dependency: `transformers`, `safetensors`)
- Convert once; runtime is zero-dep C
- INT8 quantization using percentile calibration (not RTN)
- Validate output against HuggingFace reference

---

### Phase E: KV Cache Optimization (Week 9)

| Feature | Description | Gain |
|---------|-------------|-----:|
| FP8 KV cache storage | Store K/V in FP8, dequant on read | 2× context length |
| Context window expansion | 4K → 8K (32 GB RAM) | Better long-document support |
| Disk KV cache swap | Offload oldest tokens to NVMe | Unlimited context (slow) |

---

### Phase F: Platform Support (Week 10)

| Platform | Work Required |
|----------|--------------|
| macOS ARM | NEON IDOT kernels, MachO build, Metal GPU? (future) |
| Windows | MSVC compatibility, Win32 thread pool, eventfd→Event handle |
| Linux | Already supported (baseline) |

---

### Phase G: Package & Release (Weeks 11-12)

#### Distribution Options

| Method | Users | Complexity |
|--------|-------|------------|
| GitHub release + binary | Developers | Low |
| Homebrew (macOS) | macOS users | Medium |
| `pip install colibri-lite` | Python users | Medium (bundles C binary) |
| Docker image | Cloud/server | Low |

#### Release Checklist

- [ ] Binary builds for Linux (x86_64, aarch64), macOS (arm64), Windows (x86_64)
- [ ] CI pipeline: build → test → benchmark → release
- [ ] Documentation: README, usage examples, FAQ
- [ ] 3+ supported models verified end-to-end
- [ ] Performance numbers published

---

## Goal-Based Decision Tree

```
What is your primary goal?
│
├─ Product (users, distribution)
│   ├─ Focus on: Model converter (Phase D) + Package (Phase G)
│   ├─ MVP: 3 models, chat + serve, single binary
│   └─ Metric: "Works out of the box" for new models
│
├─ Educational (learning, research)
│   ├─ Focus on: Speculative decoding research + KV cache compression
│   ├─ Add: MTP acceptance rate optimization, draft model experiments
│   └─ Metric: Novel contributions, accepted improvements
│
├─ Engineering (craft, clean code)
│   ├─ Focus on: Refactoring (Phase A) + Infrastructure (Phase C)
│   ├─ Add: Full test suite, static analysis, fuzzing
│   └─ Metric: Code quality, test coverage, maintainability
│
└─ Portfolio (demonstration, career)
    ├─ Focus on: All phases, but document everything
    ├─ Add: Blog posts, architecture diagrams, performance analysis
    └─ Metric: "I built an LLM inference engine from scratch"
```

---

## Key Design Decisions

### D1. C over Rust/C++

- **Choice:** C11 (same as Colibri)
- **Why:** Zero runtime deps; easiest FFI; simplest build; every platform has a C compiler
- **Cost:** Manual memory management, no RAII
- **Mitigation:** Arena allocators, strict ownership conventions

### D2. Single Thread Pool, Not Per-Layer Threads

- **Choice:** 1 PILOT thread + N WORKER threads (total configurable)
- **Why:** Prevents oversubscription; matches hardware concurrency
- **Cost:** Less parallelism on very deep models
- **Mitigation:** Pipeline overlap compensates

### D3. Generic Model Interface, Not Per-Model Specialization

- **Choice:** Abstract model config struct → all models share same code paths
- **Why:** Enables multi-model support with single engine binary
- **Cost:** Some model-specific optimizations impossible
- **Mitigation:** Hot-path dispatch via function pointers (not virtual)

### D4. Python for Converter Only

- **Choice:** Converter is Python; runtime is C
- **Why:** Python ecosystem access (`transformers`, `safetensors`); separation of concerns
- **Cost:** Python dependency for conversion step
- **Mitigation:** Conversion is one-time; binary is self-contained

---

## Measurement Plan

### Build Metrics
- Compilation time: target < 10 seconds
- Binary size: target < 2 MB (stripped)
- Lines of code per module: target < 500

### Runtime Metrics
- Cold decode throughput: target ≥ 10 tok/s
- Warm decode throughput: target ≥ 20 tok/s
- Memory RSS: target < 16 GB (1-hour session)
- GPU VRAM: target < 22 GB (24 GB cap)
- First-token latency: target < 5 seconds

### Quality Metrics
- Output bit-exact after refactoring (Phase A)
- INT8 perplexity within 1% of FP16 reference
- No crashes in 24-hour stress test
- Memory leak: 0 bytes over 1-hour session

---

## Contingency Plans

### If Converter is Too Hard
- **Fallback:** Support only GLM-5.2 and DeepSeekMoE (same MoE architecture)
- **Trade:** Fewer models, but done correctly

### If Performance Targets Not Met
- **Fallback:** Accept 5-8 tok/s cold decode (matches original Colibri)
- **Trade:** Slower but stable; optimize later

### If CUDA Backend Takes Too Long
- **Fallback:** CPU-only release; GPU is Phase 4
- **Trade:** Slower warm decode (no PCIe bottleneck), but functional

### If Page Cache Behavior is Unfixable on Windows
- **Fallback:** Larger expert cache in GPU VRAM; smaller active model
- **Trade:** Reduce usable model size; maintain stability

---

## Immediate Next Steps

1. **Create repository structure** — `c/`, `python/`, `docs/`, `tests/`
2. **Copy initial files from Colibri** — start with `tok.h`, `grammar.h`, `json.h`
3. **Write build system** — `Makefile` with modular targets
4. **Begin `glm.c` decomposition** — extract `model.h` (config structs) first
5. **Verification harness** — capture current `glm` output for regression testing
