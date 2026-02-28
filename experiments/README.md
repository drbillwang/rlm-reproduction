# Experiment Scripts

This folder contains all experiment drivers for reproducing the RLM results on S-NIAH (RULER) and OOLONG (trec_coarse) benchmarks.

## Quick Start

### 1. Install dependencies

```bash
# From repo root
cd rlm && pip install -e . && cd ..
pip install datasets python-dotenv openai
```

### 2. Configure API keys

```bash
cp .env.template .env
# Edit .env — fill in DEEPSEEK_API_KEY, KIMI_API_KEY, or OPENAI_API_KEY
```

### 3. Run experiments

```bash
# --- S-NIAH (RULER) — 20 samples each ---
python run_ruler_plain.py       # Base LLM (no RLM)
python run_ruler_baseline.py    # RLM depth=1
python run_ruler_depth2.py      # RLM depth=2

# --- OOLONG (trec_coarse) — 20 samples each ---
python run_oolong_plain.py      # Base LLM (no RLM)
python run_oolong_depth1.py     # RLM depth=1
python run_oolong_depth2.py     # RLM depth=2

# --- Compare all results ---
python compare_results.py
```

Results are saved to `results/` as JSON files.

---

## Experimental Design

### Research Question

> Does the RLM framework improve LLM performance on long-context tasks? How does recursion depth affect accuracy, latency, and cost?

### Benchmarks

| Benchmark | Complexity | Samples | Source |
|-----------|:---:|:---:|--------|
| **S-NIAH** (RULER) | O(1) retrieval | 20 | `data/ruler/niah_single_2/validation.jsonl` |
| **OOLONG** (trec_coarse) | O(N) reasoning | 20 | HuggingFace `oolongbench/oolong-synth` (validation) |

### Conditions

| Condition | Description |
|-----------|-------------|
| **Base LLM** | Direct single LLM call — no RLM, no REPL |
| **RLM (depth=1)** | REPL + code generation; sub-calls use plain LLM |
| **RLM (depth=2)** | Sub-calls can spawn their own REPL environments |

### Metrics

| Benchmark | Metric |
|-----------|--------|
| S-NIAH | Exact-match accuracy |
| OOLONG | Numerical: `max(0, 1 − 0.75·|y − ŷ|)` ; Others: exact match |

---

## File Overview

### Core Experiment Drivers

| File | Description |
|------|-------------|
| `run_ruler_experiment.py` | S-NIAH — RLM experiment driver (configurable depth) |
| `run_ruler_plain_llm.py` | S-NIAH — Base LLM driver (no RLM) |
| `run_oolong_experiment.py` | OOLONG — RLM experiment driver (configurable depth) |
| `run_oolong_plain_llm.py` | OOLONG — Base LLM driver (no RLM) |
| `compare_results.py` | Aggregate and compare all runs |
| `batch_run_experiments.py` | Run all conditions in batch |

### Convenience Entry Points

| File | Condition |
|------|-----------|
| `run_ruler_plain.py` | S-NIAH — Base LLM |
| `run_ruler_baseline.py` | S-NIAH — RLM depth=1 |
| `run_ruler_depth2.py` | S-NIAH — RLM depth=2 |
| `run_oolong_plain.py` | OOLONG — Base LLM |
| `run_oolong_depth1.py` | OOLONG — RLM depth=1 |
| `run_oolong_depth2.py` | OOLONG — RLM depth=2 |

### Shell Scripts

| File | Description |
|------|-------------|
| `run.sh` | General run script |
| `run_deepseek_v32_all.sh` | Run all DeepSeek v3.2 experiments |

---

## Output Format

Each experiment produces a JSON file in `results/` containing:

- Aggregate accuracy and average score
- Average execution time per task
- Total input/output tokens and approximate cost (USD)
- Per-example: response text, score, tokens, cost, and execution time

Pre-computed results for both DeepSeek v3.2 and Kimi K2 are available in `../reproduction_results/`.

---

## Cost Notes

- **Base LLM** experiments are cheapest (single API call per sample)
- **RLM depth=1** adds REPL overhead (multiple API calls per sample)
- **RLM depth=2** is the most expensive (recursive sub-calls can spawn further sub-calls)
- If you hit API rate limits, add `time.sleep()` between calls or reduce batch size
