# W2D1 Extra Lab — Memory Budget Solver

Computes, from real model architecture fields (not marketing names), the
largest model+precision combination that fits a given GPU memory budget.

**Scenario:** 16 GB budget, 4 concurrent users, 4096 context tokens.

## Files
- `solve.py` — derives parameter counts and KV-cache cost for 5 models
  (Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B/3B) across fp16/int8/int4, and writes
  `budget_solution.json`.
- `budget_solution.json` — the computed solution (15 model×precision rows).
- `verify.py` — independent green-check verifier; recomputes everything from
  scratch and compares.

## Run
```bash
python solve.py      # regenerates budget_solution.json
python verify.py      # GREEN CHECK: PASS
```

## Result
Best fitting combination: **Llama-3.2-3B-Instruct / fp16** (~9.80 GB total).
