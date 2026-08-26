import json

CATALOG = {
    "Qwen2.5-0.5B-Instruct": {"hidden_size": 896,  "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "intermediate_size": 4864,  "vocab_size": 151936, "tie_word_embeddings": True},
    "Qwen2.5-1.5B-Instruct": {"hidden_size": 1536, "num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2, "intermediate_size": 8960,  "vocab_size": 151936, "tie_word_embeddings": True},
    "Qwen2.5-3B-Instruct":   {"hidden_size": 2048, "num_hidden_layers": 36, "num_attention_heads": 16, "num_key_value_heads": 2, "intermediate_size": 11008, "vocab_size": 151936, "tie_word_embeddings": True},
    "Llama-3.2-1B-Instruct": {"hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 8192,  "vocab_size": 128256, "tie_word_embeddings": True},
    "Llama-3.2-3B-Instruct": {"hidden_size": 3072, "num_hidden_layers": 28, "num_attention_heads": 24, "num_key_value_heads": 8, "intermediate_size": 8192,  "vocab_size": 128256, "tie_word_embeddings": True},
}
PRECISIONS = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
SCENARIO = {"budget_gb": 16, "concurrent_users": 4, "context_tokens": 4096}
OVERHEAD_GB = 1.5

def count_params(cfg):
    h, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    head_dim = h // cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"] * head_dim
    attn = h * h + h * kv + h * kv + h * h
    mlp = 3 * h * cfg["intermediate_size"]
    embeds = (1 if cfg.get("tie_word_embeddings") else 2) * cfg["vocab_size"] * h
    return L * (attn + mlp) + embeds

def kv_bytes_per_token(cfg, kv_dtype_bytes=2):
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * head_dim * kv_dtype_bytes

rows = []
for name, cfg in CATALOG.items():
    p = count_params(cfg)
    kv_gb = (kv_bytes_per_token(cfg) * SCENARIO["context_tokens"] * SCENARIO["concurrent_users"] / 1e9)
    for prec, bpp in PRECISIONS.items():
        w = p * bpp / 1e9
        total = w + kv_gb + OVERHEAD_GB
        rows.append({
            "model": name,
            "precision": prec,
            "params": p,
            "weight_gb": round(w, 2),
            "kv_gb": round(kv_gb, 2),
            "total_gb": round(total, 2),
            "fits": total <= SCENARIO["budget_gb"],
        })

fitting = [r for r in rows if r["fits"]]
best_row = max(fitting, key=lambda r: r["params"])

solution = {
    "budget_gb": SCENARIO["budget_gb"],
    "concurrent_users": SCENARIO["concurrent_users"],
    "context_tokens": SCENARIO["context_tokens"],
    "best": {"model": best_row["model"], "precision": best_row["precision"]},
    "all_combinations": [
        {k: v for k, v in r.items() if k != "params"} for r in rows
    ],
}

with open("/home/claude/budget_solution.json", "w") as f:
    json.dump(solution, f, indent=2)

print(json.dumps(solution, indent=2))
