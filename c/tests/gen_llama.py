"""Generate a tiny synthetic LLaMA model in safetensors format for end-to-end testing."""
import struct, json, hashlib, os

CFG = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 512,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "eos_token_id": 2,
    "head_dim": 16,
}

N = CFG["num_hidden_layers"]
H = CFG["num_attention_heads"]
K = CFG["num_key_value_heads"]
D = CFG["hidden_size"]
I = CFG["intermediate_size"]
V = CFG["vocab_size"]
hd = D // H  # 16

def tensor_name(layer, name):
    return f"model.layers.{layer}.{name}"

# All tensors we need
entries = []

# Embedding
entries.append(("model.embed_tokens.weight", (V, D)))
# LM head
entries.append(("lm_head.weight", (V, D)))
# Final norm
entries.append(("model.norm.weight", (D,)))

for i in range(N):
    entries.append((tensor_name(i, "input_layernorm.weight"), (D,)))
    entries.append((tensor_name(i, "post_attention_layernorm.weight"), (D,)))
    entries.append((tensor_name(i, "self_attn.q_proj.weight"), (H * hd, D)))
    entries.append((tensor_name(i, "self_attn.k_proj.weight"), (K * hd, D)))
    entries.append((tensor_name(i, "self_attn.v_proj.weight"), (K * hd, D)))
    entries.append((tensor_name(i, "self_attn.o_proj.weight"), (D, H * hd)))
    entries.append((tensor_name(i, "mlp.gate_proj.weight"), (I, D)))
    entries.append((tensor_name(i, "mlp.up_proj.weight"), (I, D)))
    entries.append((tensor_name(i, "mlp.down_proj.weight"), (D, I)))

# Build data
rng = __import__("random").Random(42)
data = bytearray()
offsets = {}
header_entries = {}

for name, shape in entries:
    numel = 1
    for s in shape:
        numel *= s
    off = len(data)
    nbytes = numel * 4  # F32
    data.extend(struct.pack(f"<{numel}f", *(rng.uniform(-1, 1) for _ in range(numel))))
    offsets[name] = (off, off + nbytes)
    header_entries[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + nbytes]}

header = json.dumps(header_entries, separators=(",", ":")).encode("utf-8")
hlen = len(header)

outdir = os.path.join(os.path.dirname(__file__), "..", "snap_llama_test")
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

print(f"Created synthetic LLaMA model in {outdir}")
print(f"  vocab_size={V}, hidden={D}, heads={H}, kv_heads={K}, layers={N}, intermediate={I}")
sz = os.path.getsize(os.path.join(outdir, "model.safetensors"))
print(f"  model.safetensors: {sz} bytes ({sz/1024/1024:.2f} MB)")
