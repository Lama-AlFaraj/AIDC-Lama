#!/usr/bin/env python3
import json, os
from typing import NoReturn

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
TOL_GB = 0.06


class _Stop(Exception):
    pass


def _fail(reason) -> NoReturn:
    print("GREEN CHECK: FAIL (%s)" % reason)
    raise _Stop()


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


def truth():
    rows = {}
    for name, cfg in CATALOG.items():
        p = count_params(cfg)
        kv_gb = (kv_bytes_per_token(cfg) * SCENARIO["context_tokens"]
                 * SCENARIO["concurrent_users"] / 1e9)
        for prec, bpp in PRECISIONS.items():
            w = p * bpp / 1e9
            total = w + kv_gb + OVERHEAD_GB
            rows[(name, prec)] = {"params": p, "weight_gb": w, "kv_gb": kv_gb,
                                  "total_gb": total,
                                  "fits": total <= SCENARIO["budget_gb"]}
    return rows


def main():
    if not os.path.isfile("budget_solution.json"):
        _fail("budget_solution.json not found next to this script; run Step 5 first")
    try:
        with open("budget_solution.json") as f:
            sol = json.load(f)
    except json.JSONDecodeError as e:
        _fail("budget_solution.json is not valid JSON: %s" % e)

    for key in ("budget_gb", "concurrent_users", "context_tokens", "best", "all_combinations"):
        if key not in sol:
            _fail("missing top-level key '%s'" % key)
    for key, want in SCENARIO.items():
        if sol[key] != want:
            _fail("scenario %s=%r; the green check runs against the canonical "
                  "scenario %r (Step 5's numbers)" % (key, sol[key], want))

    rows = sol["all_combinations"]
    if not isinstance(rows, list) or len(rows) != len(CATALOG) * len(PRECISIONS):
        _fail("all_combinations must hold %d rows (5 models x 3 precisions), got %s"
              % (len(CATALOG) * len(PRECISIONS), len(rows) if isinstance(rows, list) else type(rows).__name__))

    t = truth()
    seen = set()
    for r in rows:
        key = (r.get("model"), r.get("precision"))
        if key not in t:
            _fail("unknown model/precision pair %r" % (key,))
        if key in seen:
            _fail("duplicate row for %r" % (key,))
        seen.add(key)
        want = t[key]
        for field in ("weight_gb", "kv_gb", "total_gb"):
            got = r.get(field)
            if not isinstance(got, (int, float)):
                _fail("%s missing numeric %s" % (key, field))
            if abs(got - want[field]) > TOL_GB:
                _fail("%s %s=%.2f, expected %.2f (formula or unit slip; "
                      "check tie_word_embeddings and GQA kv width first)"
                      % (key, field, got, want[field]))
        if bool(r.get("fits")) != want["fits"]:
            _fail("%s fits=%r, expected %r" % (key, r.get("fits"), want["fits"]))

    best = sol["best"]
    fitting = {k: v for k, v in t.items() if v["fits"]}
    if not fitting:
        _fail("verifier bug: canonical scenario should have fitting combos")
    want_key = max(fitting, key=lambda k: fitting[k]["params"])
    if best is None:
        _fail("best is null but %s/%s fits the budget" % want_key)
    if (best.get("model"), best.get("precision")) != want_key:
        _fail("best is %s/%s; the largest fitting combination is %s/%s"
              % (best.get("model"), best.get("precision"), *want_key))

    print("checked %d combinations against an independent recomputation" % len(rows))
    print("GREEN CHECK: PASS")


if __name__ == "__main__":
    try:
        main()
    except _Stop:
        raise SystemExit(1)
