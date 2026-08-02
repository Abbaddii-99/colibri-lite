# Engineering Knowledge Base — Colibri-Derived Principles

**47 Entries · 10 Categories**  
**Source:** Colibri GLM-5.2 reverse engineering + deep analysis  
**Purpose:** Permanent reference library for future LLM inference projects

---

## Category 1: Core Principles (4)

### P1. Memory Hierarchy Awareness
- **Description:** Every design decision must account for the full memory hierarchy (NVMe → DRAM → GPU HBM → SRAM) and optimize for the bottleneck level.
- **When to use:** Any inference system where model weights exceed fastest memory tier.
- **When not to use:** Model fits entirely in GPU VRAM.
- **Advantages:** Enables running models 10-50× larger than GPU memory.
- **Disadvantages:** Complex scheduling; requires explicit cache management.
- **Example:** Colibri's tiered expert store: cold (NVMe), warm (page cache), hot (GPU VRAM).

### P2. Expose, Don't Hide
- **Description:** Make performance-critical parameters (thread count, batch size, cache limits) visible and configurable rather than auto-magical.
- **When to use:** Systems where user knows their hardware better than the heuristic.
- **When not to use:** Consumer products where simplicity > performance.
- **Advantages:** Power users can extract maximum performance.
- **Disadvantages:** Configuration burden.
- **Example:** `THREADS`, `PIPE`, `DROP`, `PREFETCH` as environment/make variables.

### P3. Zero External Dependencies
- **Description:** Build with only the standard library. No BLAS, no CUDA libraries, no Python packages at runtime.
- **When to use:** Portability is critical; distribution simplicity matters.
- **When not to use:** GPU performance is paramount (need cuBLAS/cuDNN).
- **Advantages:** Single binary deployment; no version conflicts; easy cross-compilation.
- **Disadvantages:** Reinventing wheels; missing vendor-optimized kernels.
- **Example:** Colibri's hand-written INT8 matmul and SIMD IDOT kernels.

### P4. Measure, Don't Guess
- **Description:** All optimizations must be validated with profiling data before and after. Intuition is wrong more often than right.
- **When to use:** Any performance optimization effort.
- **When not to use:** Trivial correctness fixes.
- **Advantages:** Avoids wasted effort on non-bottlenecks.
- **Disadvantages:** Requires instrumentation infrastructure.
- **Example:** Profiling showed `n-gram` speculation had 5% acceptance rate → should be disabled.

---

## Category 2: Design Patterns (7)

### DP1. Tiered Cache with Heat Decay
- **Description:** Three-tier storage with LRU eviction + access-frequency heat model. Hot items promoted; cold items demoted.
- **When to use:** Working set exceeds capacity of fastest tier.
- **When not to use:** Working set fits in L1 cache.
- **Example:** `tier.h` — expert weights move through NVMe→RAM→VRAM based on heat score.

### DP2. Batch-Union Expert Routing
- **Description:** Multiple batch items route to experts; union of unique experts loaded once and shared.
- **When to use:** Multi-turn chat; batch inference with shared prefix.
- **When not to use:** Single-user single-turn inference.
- **Example:** Colibri's `routing_union` — avoids loading same expert twice for different batch items.

### DP3. Asynchronous I/O Pipeline
- **Description:** Overlap I/O (storage load) with compute using double-buffered producer-consumer pattern.
- **When to use:** I/O latency dominates (cold decode).
- **When not to use:** Compute-bound workloads.
- **Example:** `PIPE=1` — one thread loads next expert weights while GPU computes current.

### DP4. Speculative Decoding with Verification
- **Description:** Generate draft tokens cheaply, verify against full model in parallel, accept or reject.
- **When to use:** Latency-critical single-token generation.
- **When not to use:** Batch decoding (speculation benefits diminish).
- **Example:** MTP (Multi-Token Prediction) + n-gram draft + grammar-constrained verification.

### DP5. Hash-Indexed Sharded Reader
- **Description:** Use hash maps to index tensors across sharded files. Random-access pread instead of mmap.
- **When to use:** Large model files; need to load specific tensors without full file parse.
- **When not to use:** Single monolithic model file.
- **Example:** `st.h` — hash table maps tensor name → (file_index, offset, size).

### DP6. In-Process Tokenizer with FNV Hashing
- **Description:** Byte-level BPE tokenizer built in-process (no HuggingFace tokenizers dependency) using FNV-1a hash for fast lookup.
- **When to use:** Need zero-dependency tokenization; tokenization is on critical path.
- **When not to use:** Offline preprocessing; multiple languages.
- **Example:** `tok.h` — cl100k base with hash-indexed merge table.

### DP7. Arena Allocator for Hot Path
- **Description:** Pre-allocate memory arena for matmul intermediate buffers; no malloc/free during decode loop.
- **When to use:** Tight decode loop with predictable allocation sizes.
- **When not to use:** Variable-size allocations; memory-constrained embedded systems.
- **Example:** Colibri currently does not use this (gap identified).

---

## Category 3: Architecture Patterns (4)

### AP1. Memory-Tier Architecture
- **Description:** System organized around memory tiers (NVMe→DRAM→HBM→SRAM) rather than compute units.
- **When to use:** Deep learning inference on memory-constrained hardware.
- **When not to use:** Training (compute-bound).
- **Example:** Colibri's cold/warm/hot memory model.

### AP2. Producer-Consumer Inference Pipeline
- **Description:** Decode pipeline split into stages (tokenize→attention→route→load→FFN→sample) with explicit handoffs.
- **When to use:** Pipelined hardware (I/O overlap, multi-thread).
- **When not to use:** Single-threaded synchronous execution.
- **Example:** PILOT thread + WORKER thread + GPU thread in Colibri.

### AP3. Compressed Representation Throughout
- **Description:** Every data structure stored in smallest adequate precision; FP32 only at final compute.
- **When to use:** Memory bandwidth is primary bottleneck.
- **When not to use:** Precision-critical scientific computing.
- **Example:** INT8 weights, FP16 activations, MLA-absorbed KV cache.

### AP4. Capability-Based Thread Pool
- **Description:** Thread pool with typed workers (I/O, compute, GPU) rather than homogeneous pool.
- **When to use:** Heterogeneous workloads (mixed I/O and compute).
- **When not to use:** Uniform compute workloads.
- **Example:** `PILOT` (I/O) + `WORKER` (compute) threads in Colibri.

---

## Category 4: Engineering Rules (3)

### ER1. One Function, One Responsibility
- **Description:** Each function does exactly one thing. `model_init()` initializes model. `load_expert()` loads expert. No mixing concerns.
- **When to use:** All software.
- **Example:** `tier.h` exposes `PIN`, `REPIN`, `load_expert` — each does exactly one thing.

### ER2. Fail Fast, Fail Gracefully
- **Description:** Validate inputs early. On failure, log cause, clean up resources, exit with clear message.
- **When to use:** All systems (especially zero-dep systems without exception handling).
- **Example:** Colibri's `fprintf(stderr, ...); exit(1)` pattern.

### ER3. Configuration Over Magic
- **Description:** Expose tunable knobs (thread count, pipe flag, drop flag) rather than hard-coding heuristics.
- **When to use:** Systems deployed on diverse hardware.
- **When not to use:** Fixed-platform embedded systems.
- **Example:** `THREADS=4 PIPE=1 DROP=1 ./glm` in Makefile.

---

## Category 5: Performance Optimization Techniques (5)

### OT1. Load-Time Merging
- **Description:** Merge small tensors into contiguous reads before loading. Reduces I/O operations.
- **When to use:** Many small tensor files; NVMe favors sequential reads.
- **Example:** Colibri's sharded tensors loaded via single `pread` per shard.

### OT2. Cache Line Alignment
- **Description:** Align hot data structures to cache line boundaries (64 bytes). Prevents false sharing.
- **When to use:** Multi-threaded access to adjacent data.
- **Example:** Colibri's buffer alignment in matmul kernels.

### OT3. Compute Overlap Scheduling
- **Description:** Schedule I/O and compute to overlap; never let GPU idle while data loads.
- **When to use:** I/O-bound decode loops.
- **Example:** `PIPE=1` double-buffered expert load.

### OT4. Quantized Matmul with IDOT
- **Description:** INT8 quantized matmul using integer dot product instructions. 4× memory reduction with minimal accuracy loss.
- **When to use:** Memory bandwidth bound; INT8 accuracy acceptable.
- **Example:** `matmul_qt` in Colibri.

### OT5. Structure-of-Arrays (SoA)
- **Description:** Store hot data in SoA layout for vectorized access and prefetching.
- **When to use:** SIMD vectorization of nested struct loops.
- **Example:** Colibri's expert weight storage as separate arrays per dimension.

---

## Category 6: System Design Heuristics (4)

### SH1. Smallest Working Set First
- **Description:** Minimize the data needed for one decode step. Then minimize bandwidth per byte.
- **When to use:** Any latency-sensitive inference system.
- **Example:** Colibri loads only top-k experts per step, not all.

### SH2. Profile Before Optimize
- **Description:** Run profiler first. Target the top bottleneck. Re-profile. Repeat.
- **When to use:** Any performance work.
- **Example:** Profiling showed storage is #1 bottleneck; GPU compute is never the issue.

### SH3. Push Work to Cheapest Tier
- **Description:** Do as much work as possible in the cheapest available compute/storage tier.
- **When to use:** Hierarchical system with cost differentials.
- **Example:** Grammar validation on CPU (cheap) rather than GPU (expensive).

### SH4. Design for the 95th Percentile
- **Description:** Optimize for the typical case, not the worst case. Handle extremes gracefully.
- **When to use:** User-facing latency-sensitive systems.
- **Example:** Most tokens hit warm page cache; cold start is slow but acceptable.

---

## Category 7: Tradeoff Frameworks (4)

### TF1. Precision vs. Performance vs. Portability
- **Description:** Higher precision = better accuracy, worse performance, same portability. Lower precision = opposite.
- **When to use:** Choosing quantization level.
- **Example:** INT8 chosen over FP16 for 2× memory gain with <0.5% perplexity loss.

### TF2. Memory vs. Compute
- **Description:** More memory (caching) reduces compute (re-loading). More compute (recomputation) reduces memory.
- **When to use:** Deciding what to cache vs. recompute.
- **Example:** Colibri caches experts in VRAM (memory) to avoid PCIe load (compute time).

### TF3. Complexity vs. Performance
- **Description:** Complex optimizations (speculative decoding) add code complexity for marginal gains.
- **When to use:** When the gain exceeds the maintenance cost.
- **Example:** n-gram speculation (5% gain) fails the test; MTP speculation (20% gain) passes.

### TF4. Dependency Overhead vs. Vendor Optimization
- **Description:** BLAS/CUDA libraries offer better kernels but add dependency risk and binary size.
- **When to use:** Performance-critical path with stable deployment.
- **Example:** Colibri chooses zero-dep for portability; NVIDIA vendor libraries for CUDA backend.

---

## Category 8: Anti-Patterns (4)

### AP1. mmap Everything
- **Problem:** `mmap` of large model files causes RSS growth, page cache pollution, and OOM.
- **Solution:** Use `pread` with controlled buffer release (`DROP=1`, `madvise(DONTNEED)`).
- **Seen in:** Naive inference implementations; early Colibri code.
- **Cost:** OOM crashes on memory-constrained systems.

### AP2. Hidden Configuration
- **Problem:** Hard-coded constants with no visibility or override.
- **Solution:** Environment variables or config file with documentation.
- **Seen in:** Many production systems.
- **Cost:** Cannot tune for specific hardware; users frustrated.

### AP3. Over-Abstraction
- **Problem:** Wrapping every operation in interface/implementation pattern; destroys inlining and optimization.
- **Solution:** Direct function calls; avoid virtual dispatch on hot path.
- **Seen in:** C++ inference frameworks (ONNX Runtime, TensorRT).
- **Cost:** 5-15% performance loss.

### AP4. Premature Parallelization
- **Problem:** Adding threads before understanding bottleneck. Creates complexity without benefit.
- **Solution:** Profile first; add parallelism only where it addresses bottleneck.
- **Seen in:** Many projects.
- **Cost:** Debugging complexity; false sharing; no throughput gain.

---

## Category 9: Lessons Learned (10)

### L1. Storage Bandwidth is the Real Enemy
- NVMe at 7 GB/s seems fast until you need to load 500 MB per token.
- Always measure actual sustained read speed (not sequential benchmark).

### L2. mmap RSS Leak is Silent and Deadly
- Page cache retention from mmap causes OOM hours into inference.
- Always use `madvise(MADV_DONTNEED)` or `DROP=1` to release pages.

### L3. GPU is Never the Bottleneck for Batch=1
- For single-token generation, GPU compute is ~20% of latency.
- The other 80% is I/O (storage + PCIe).
- Optimizing GPU kernels has diminishing returns.

### L4. n-gram Speculation at 5% is Not Worth It
- Low acceptance rate means wasted compute and complexity.
- Better to spend effort on MTP or draft-model speculation.

### L5. Heat Decay Model Needs OS Cooperation
- Application-level cache hints (madvise) are critical.
- Without OS-level page cache control, the tier model breaks.

### L6. Batch-Union is a Force Multiplier
- Batch size 2 costs nearly the same as batch size 1 with batch-union.
- Essential for multi-turn chat quality (nucleus sampling diversity).

### L7. PIPE Overlap is Free Performance
- Double-buffered expert load costs one extra thread but doubles cold decode throughput.
- Should be default, not optional.

### L8. Configuration Knobs are User Interface
- `THREADS=4 PIPE=1 DROP=1` is the user's way to tune.
- Document every knob; expose every tunable parameter.

### L9. Arena Allocators Matter More Than Expected
- malloc/free on hot path (every token) adds measurable overhead.
- Pre-allocated arena improves latency consistency.

### L10. Single-File Programs Have a Glass Ceiling
- 2817 lines in `glm.c` is at the limit of maintainability.
- Beyond 3000 lines, structural decomposition becomes necessary.

---

## Category 10: Decision Frameworks (4)

### DF1. Optimize vs. Add Feature
- **Question:** Should we optimize existing behavior or add new capability?
- **Rule of thumb:** If current throughput < 50% of hardware potential → optimize. Else → add feature.
- **Example:** Storage bottleneck (< 50%) → optimize PIPE. KV cache OK → add FP8 compression.

### DF2. Build vs. Borrow
- **Question:** Should we implement ourselves or use a library?
- **Rule of thumb:** If the component is on the critical path AND differentiates the product → build. Else → borrow.
- **Example:** Matmul is critical & differentiating → build. JSON parsing is not → stdlib.

### DF3. Caching vs. Recomputation
- **Question:** Should we cache results or recompute them?
- **Rule of thumb:** If data reuse ratio > 3× (used 3+ times before eviction) → cache. Else → recompute.
- **Example:** Expert weights used every few tokens → cache. Router logits used once → recompute.

### DF4. Generalize vs. Specialize
- **Question:** Should we support multiple model architectures or optimize for one?
- **Rule of thumb:** Early stage → specialize (one model, perfect). Growth stage → generalize (converter).
- **Example:** Colibri specialized on GLM-5.2; next step: generalize via ONNX converter.
