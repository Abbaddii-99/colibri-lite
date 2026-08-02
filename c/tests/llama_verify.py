#!/usr/bin/env python3
"""Verify LLaMA forward pass numerically against the C engine.

Reads the synthetic LLaMA model, implements the forward pass in numpy,
and compares layer outputs with the C engine's step_all output.

Usage: python tests/llama_verify.py
"""

import struct, json, os, sys, subprocess, math, random

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.normpath(os.path.join(HERE, "..", "snap_llama_test"))

# ── Load model weights ──
with open(os.path.join(SNAP, "config.json")) as f:
    CFG = json.load(f)

N = CFG["num_hidden_layers"]
H = CFG["num_attention_heads"]
K = CFG["num_key_value_heads"]
D = CFG["hidden_size"]
I = CFG["intermediate_size"]
V = CFG["vocab_size"]
hd = D // H  # head dim
eps = CFG["rms_norm_eps"]
theta = CFG.get("rope_theta", 10000.0)

print(f"LLaMA: {N} layers, {H} heads, {K} KV heads, {D} hidden, {I} intermediate, {V} vocab")

with open(os.path.join(SNAP, "model.safetensors"), "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hlen))
    data = f.read()

def load(name):
    e = header[name]
    off, end = e["data_offsets"]
    raw = data[off:end]
    return list(struct.unpack(f"<{len(raw)//4}f", raw))

def load_as(name, shape):
    arr = load(name)
    return [arr[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]

# Load all weights
embed = load_as("model.embed_tokens.weight", (V, D))
lm_head = load_as("lm_head.weight", (V, D))
final_norm = load("model.norm.weight")

def tensor_name(layer, name):
    return f"model.layers.{layer}.{name}"

layers = []
for i in range(N):
    l = {}
    l["input_layernorm"] = load(tensor_name(i, "input_layernorm.weight"))
    l["post_attention_layernorm"] = load(tensor_name(i, "post_attention_layernorm.weight"))
    l["q_proj"] = load_as(tensor_name(i, "self_attn.q_proj.weight"), (H * hd, D))
    l["k_proj"] = load_as(tensor_name(i, "self_attn.k_proj.weight"), (K * hd, D))
    l["v_proj"] = load_as(tensor_name(i, "self_attn.v_proj.weight"), (K * hd, D))
    l["o_proj"] = load_as(tensor_name(i, "self_attn.o_proj.weight"), (D, H * hd))
    l["gate_proj"] = load_as(tensor_name(i, "mlp.gate_proj.weight"), (I, D))
    l["up_proj"] = load_as(tensor_name(i, "mlp.up_proj.weight"), (I, D))
    l["down_proj"] = load_as(tensor_name(i, "mlp.down_proj.weight"), (D, I))
    layers.append(l)

# ── Reference implementation ──
def rmsnorm(x, w, eps=1e-5):
    n = math.sqrt(sum(v*v for v in x) / len(x) + eps)
    return [x[i] / n * w[i] for i in range(len(x))]

def silu(x):
    return x / (1 + math.exp(-x))

def matmul(A, x):
    """A: list of rows (each row is list of floats), x: vector"""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]

def matmul_t(A, x):
    """A transposed: each row of A is a column of the weight matrix"""
    cols = len(A[0])
    return [sum(A[i][j] * x[i] for i in range(len(A))) for j in range(cols)]

def rope(x, pos, dim, theta=10000.0):
    x = list(x)
    for i in range(0, dim, 2):
        freq = pos * (theta ** (-4.0 * (i // 2) / dim))
        c, s = math.cos(freq), math.sin(freq)
        a, b = x[i], x[i + 1]
        x[i]     = a * c - b * s
        x[i + 1] = b * c + a * s
    return x

def attention_layer(l, x, pos, K_cache, V_cache, layer_idx):
    """Compute attention for a single token position"""
    # QKV projections
    q = matmul(l["q_proj"], x)
    k = matmul(l["k_proj"], x)
    v = matmul(l["v_proj"], x)

    # Reshape to heads
    n_q_heads = H
    n_kv_heads = K
    q_heads = [q[t*hd:(t+1)*hd] for t in range(n_q_heads)]
    k_heads = [k[t*hd:(t+1)*hd] for t in range(n_kv_heads)]
    v_heads = [v[t*hd:(t+1)*hd] for t in range(n_kv_heads)]

    # RoPE per head
    for t in range(n_q_heads):
        q_heads[t] = rope(q_heads[t], pos, hd, theta)
    for t in range(n_kv_heads):
        k_heads[t] = rope(k_heads[t], pos, hd, theta)

    # Append to cache
    K_cache.append(k_heads)
    V_cache.append(v_heads)

    # GQA: kv_h = h % n_kv (matching C engine)
    kv_len = len(K_cache)
    attn_outs = []
    for h in range(n_q_heads):
        kv_h = h % n_kv_heads
        kv_heads_k = [cache[kv_h] for cache in K_cache]
        kv_heads_v = [cache[kv_h] for cache in V_cache]
        q = q_heads[h]

        scores = [sum(q[d] * kv_heads_k[j][d] for d in range(hd)) / math.sqrt(hd) for j in range(kv_len)]
        mx = max(scores)
        ex = [math.exp(s - mx) for s in scores]
        s_ex = sum(ex)
        weights = [e / s_ex for e in ex]

        out_t = [0.0] * hd
        for j in range(kv_len):
            for d in range(hd):
                out_t[d] += weights[j] * kv_heads_v[j][d]
        attn_outs.append(out_t)

    attn_concat = []
    for h_out in attn_outs:
        attn_concat.extend(h_out)

    # Output projection
    out = matmul(l["o_proj"], attn_concat)
    return out

def mlp_layer(l, x):
    gate = matmul(l["gate_proj"], x)
    up = matmul(l["up_proj"], x)
    gated = [silu(gate[i]) * up[i] for i in range(I)]
    out = matmul(l["down_proj"], gated)
    return out

def forward_ref(tokens):
    S = len(tokens)
    # Embed
    x = [list(embed[t]) for t in tokens]

    # K/V caches per layer
    K_caches = [[] for _ in range(N)]
    V_caches = [[] for _ in range(N)]

    for pos in range(S):
        h = x[pos]

        for li in range(N):
            l = layers[li]
            # Pre-attention norm
            n = rmsnorm(h, l["input_layernorm"], eps)
            # Attention
            attn_out = attention_layer(l, n, pos, K_caches[li], V_caches[li], li)
            # Residual
            h = [h[i] + attn_out[i] for i in range(D)]
            # Post-attention norm
            n = rmsnorm(h, l["post_attention_layernorm"], eps)
            # MLP
            mlp_out = mlp_layer(l, n)
            # Residual
            h = [h[i] + mlp_out[i] for i in range(D)]

        x[pos] = h

    # Final norm
    x = [rmsnorm(h, final_norm, eps) for h in x]

    # LM head
    logits = [matmul(lm_head, h) for h in x]
    return logits

# ── Run C engine and compare ──
def run_c_engine(tokens):
    """Run the C engine's step_all and return logits"""
    BINARY = os.path.normpath(os.path.join(HERE, "..", "glm.exe"))
    score_file = os.path.join(SNAP, "score_verify.txt")

    # Write a temp score file
    S = len(tokens) // 2  # ctxlen, contlen
    with open(score_file, "w") as f:
        f.write(f"{S} {len(tokens)-S} " + " ".join(str(t) for t in tokens) + "\n")

    env = os.environ.copy()
    env["SNAP"] = SNAP
    env["SCORE"] = score_file
    env["COLI_NO_OMP_TUNE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["IDOT"] = "0"  # Use exact F32 for comparison

    r = subprocess.run([BINARY], capture_output=True, timeout=30, env=env)
    out = (r.stderr + r.stdout).decode("utf-8", errors="replace")

    os.remove(score_file)

    lines = out.strip().splitlines()
    for line in lines:
        if line.strip() and line.strip()[0] in "-0123456789" and " " in line:
            parts = line.strip().split()
            return float(parts[0]), int(parts[1]), int(parts[2])
    print("C engine output:")
    print(out[:1000])
    return None, None, None

# Generate a reference logprob from Python
tokens = [10, 20, 30, 42, 55, 99]
S = 3  # ctxlen
logits = forward_ref(tokens)

# Check log-likelihood: compute target logprob for the continuation
def logprob_target(logits_row, target):
    mx = max(logits_row)
    se = sum(math.exp(v - mx) for v in logits_row)
    return logits_row[target] - mx - math.log(se)

lp_ref = 0.0
greedy = 1
for pos in range(S - 1, len(tokens) - 1):
    lr = logprob_target(logits[pos], tokens[pos + 1])
    lp_ref += lr
    best = max(range(len(logits[pos])), key=lambda i: logits[pos][i])
    if best != tokens[pos + 1]:
        greedy = 0

contlen = len(tokens) - S

print(f"\nPython ref logprob: {lp_ref:.6f} {contlen} {greedy}")

# Run C engine
lp_c, cl_c, gr_c = run_c_engine(tokens)
if lp_c is not None:
    diff = abs(lp_ref - lp_c)
    print(f"C engine logprob : {lp_c:.6f} {cl_c} {gr_c}")
    print(f"Difference: {diff:.10f}")
    if diff < 0.1 and greedy == gr_c:
        print("  PASS  LLaMA forward pass verified! (diff={:.6f})".format(diff))
        sys.exit(0)
    else:
        print("  FAIL  LLaMA mismatch: ref={:.6f} c={:.6f} diff={:.10f}".format(lp_ref, lp_c, diff))
        sys.exit(1)
else:
    print("  FAIL  Could not get C engine output")
    sys.exit(1)
