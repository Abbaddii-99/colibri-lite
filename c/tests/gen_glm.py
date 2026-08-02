"""Generate a tiny synthetic GLM model in safetensors format for end-to-end testing."""
import struct, json, os, random

CFG = {
    "model_type": "glm",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 32,
    "vocab_size": 512,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "eos_token_id": 2,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "first_k_dense_replace": 0,
    "q_lora_rank": 8,
    "kv_lora_rank": 8,
    "qk_nope_head_dim": 6,
    "qk_rope_head_dim": 10,
    "v_head_dim": 8,
    "n_shared_experts": 2,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "index_topk": 0,
    "index_n_heads": 0,
    "index_head_dim": 0,
}

N = CFG["num_hidden_layers"]
H = CFG["num_attention_heads"]
D = CFG["hidden_size"]
V = CFG["vocab_size"]
ql = CFG["q_lora_rank"]
kl = CFG["kv_lora_rank"]
qn = CFG["qk_nope_head_dim"]
qr = CFG["qk_rope_head_dim"]
vh = CFG["v_head_dim"]
qh = qn + qr
mi = CFG["moe_intermediate_size"]
ne = CFG["n_routed_experts"]
ns = CFG["n_shared_experts"]

rng = random.Random(42)

def rand(*shape):
    numel = 1
    for s in shape:
        numel *= s
    return struct.pack(f"<{numel}f", *(rng.uniform(-1, 1) for _ in range(numel)))

def tensor_name(layer, name):
    return f"model.layers.{layer}.{name}"

entries = []

entries.append(("model.embed_tokens.weight", V, D))
entries.append(("lm_head.weight", V, D))
entries.append(("model.norm.weight", D,))

sI = ns * mi

for i in range(N):
    entries.append((tensor_name(i, "input_layernorm.weight"), D,))
    entries.append((tensor_name(i, "post_attention_layernorm.weight"), D,))
    entries.append((tensor_name(i, "self_attn.q_a_proj.weight"), ql, D))
    entries.append((tensor_name(i, "self_attn.q_a_layernorm.weight"), ql,))
    entries.append((tensor_name(i, "self_attn.q_b_proj.weight"), H * qh, ql))
    entries.append((tensor_name(i, "self_attn.kv_a_proj_with_mqa.weight"), kl + qr, D))
    entries.append((tensor_name(i, "self_attn.kv_a_layernorm.weight"), kl,))
    entries.append((tensor_name(i, "self_attn.kv_b_proj.weight"), H * (qn + vh), kl))
    entries.append((tensor_name(i, "self_attn.o_proj.weight"), D, H * vh))
    entries.append((tensor_name(i, "mlp.gate.weight"), ne, D))
    entries.append((tensor_name(i, "mlp.gate.e_score_correction_bias"), ne,))
    entries.append((tensor_name(i, "mlp.shared_experts.gate_proj.weight"), sI, D))
    entries.append((tensor_name(i, "mlp.shared_experts.up_proj.weight"), sI, D))
    entries.append((tensor_name(i, "mlp.shared_experts.down_proj.weight"), D, sI))
    for e in range(ne):
        entries.append((tensor_name(i, f"mlp.experts.{e}.gate_proj.weight"), mi, D))
        entries.append((tensor_name(i, f"mlp.experts.{e}.up_proj.weight"), mi, D))
        entries.append((tensor_name(i, f"mlp.experts.{e}.down_proj.weight"), D, mi))

data = bytearray()
header_entries = {}

for name, *shape in entries:
    packed = rand(*shape)
    off = len(data)
    nbytes = len(packed)
    data.extend(packed)
    header_entries[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + nbytes]}

header = json.dumps(header_entries, separators=(",", ":")).encode("utf-8")
hlen = len(header)

outdir = os.path.join(os.path.dirname(__file__), "..", "snap_glm_test")
os.makedirs(outdir, exist_ok=True)

with open(os.path.join(outdir, "model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", hlen))
    f.write(header)
    f.write(data)

with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(CFG, f, indent=2)

# Create a minimal tokenizer.json for testing
V = CFG["vocab_size"]
tokenizer = {
    "model": {
        "vocab": {chr(i): i for i in range(min(V, 256))},
        "merges": []
    },
    "added_tokens": [
        {"content": "<|endoftext|>", "id": CFG["eos_token_id"], "special": True},
        {"content": "<|pad|>", "id": 0, "special": True},
        {"content": "<|user|>", "id": 1, "special": True},
        {"content": "<|assistant|>", "id": 3, "special": True},
    ]
}
# Add remaining vocab as byte tokens
for i in range(256, V):
    tokenizer["model"]["vocab"][f"<|byte_{i}|>"] = i

with open(os.path.join(outdir, "tokenizer.json"), "w") as f:
    json.dump(tokenizer, f)

print(f"Created synthetic GLM model in {outdir}")
print(f"  vocab={V}, hidden={D}, heads={H}, layers={N}")
print(f"  q_lora={ql}, kv_lora={kl}, qk_nope={qn}, qk_rope={qr}, v_head={vh}")
print(f"  experts={ne}, topk=2, shared={ns}, moe_inter={mi}")
sz = os.path.getsize(os.path.join(outdir, "model.safetensors"))
print(f"  model.safetensors: {sz} bytes ({sz/1024/1024:.2f} MB)")
