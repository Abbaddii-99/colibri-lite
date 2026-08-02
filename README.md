# Colibri-Lite

Modular, cross-architecture inference engine for **GLM (MLA+MoE)** and **LLaMA (GQA+SwiGLU)** models.  
Designed for CPU (OpenMP) with optional CUDA backend. Bit-identical to original monolithic GLM engine.

## Features

| Capability | Details |
|------------|---------|
| **Architectures** | GLM (MLA + MoE), LLaMA / Mistral / Qwen2 (GQA + SwiGLU) |
| **Quantization** | FP32, INT8, INT4 — runtime dispatch via `IDOT` env var |
| **Hardware** | CPU (OpenMP), optional CUDA (requires nvcc) |
| **Context** | Up to 131k tokens, KV cache to disk on overflow |
| **Decoding** | Speculative: n-gram, MTP, grammar-guided drafts |
| **Serving** | HTTP server mode (`SERVE=1`) |

## Verified Correctness

| Path | Verification |
|------|--------------|
| **GLM** | Bit-identical to original monolithic `glm.c` across all settings (IDOT=0/1, OMP=1/8, S=1..4) |
| **LLaMA** | Numerical match vs independent Python reference (0.04 nat tolerance) |
| **Tests** | 21 tests: smoke + regression + edge + generation |

## Quick Start

```bash
# Build
cd c && make -j4

# Run all tests
make test

# Generate synthetic models for testing
python tests/gen_glm.py
python tests/gen_llama.py

# Score a file
SNAP=snap_llama_test SCORE=snap_llama_test/score_input.txt ./glm

# Generate text
SNAP=snap_llama_test PROMPT="Hello world" NGEN=50 ./glm

# Use real model (safetensors + tokenizer.json in model dir)
SNAP=/path/to/model PROMPT="Hello" NGEN=100 ./glm
```

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SNAP` | — | **Required**: model directory |
| `SCORE` | — | Token file for scoring |
| `PROMPT` | — | Text prompt for generation |
| `NGEN` | 64 | Tokens to generate |
| `IDOT` | 1 | 0=FP32, 1=INT8, 2=INT4 |
| `OMP_NUM_THREADS` | all | OpenMP threads |
| `CACHE` | 64 | Expert cache per layer |
| `SERVE` | 0 | HTTP server mode |

## Architecture Detection

Auto-detected from `config.json`:
```json
{ "model_type": "glm" }      // or "llama", "mistral", "qwen2"
{ "arch": "glm" }            // alternative field
```

## Project Structure

```
c/
├── src/           # Core engine (14 .c files)
├── include/       # Headers
├── tests/         # Python test suite (21 tests)
├── Makefile       # Build + test targets
└── README.md
```

## CI

GitHub Actions builds on **Windows (MSYS2/MinGW)** and **Linux (Ubuntu)**:
- Compiles modular engine
- Compiles original monolithic `glm.c` for regression
- Runs full test suite

## License

MIT — see [LICENSE](LICENSE) if present.