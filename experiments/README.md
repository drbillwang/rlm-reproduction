# RLM experiment scripts

This folder contains the scripts I use to reproduce key RLM experiments on
RULER (S-NIAH) and OOLONG (trec_coarse).

## Experiment design (following the paper)

### Research question

> Does the RLM framework improve LLM performance on long-context tasks
> compared to using the same LLM directly?

### S-NIAH (RULER)
> "Following the single needle-in-the-haystack task in RULER, we consider a set of 50 single tasks that require finding a specific phrase or number in a large set of unrelated text. Here, the information being sought scales as O(1) with respect to input length."

- **Samples**: 50
- **Task**: single needle-in-a-haystack retrieval
- **Metric**: accuracy (exact match of the needle)

### OOLONG (trec_coarse)
> "We focus specifically on the trec_coarse split, a set of 50 tasks over a dataset of questions with semantic labels. Each task requires using nearly all entries of the dataset, and therefore scales linearly in processing complexity relative to the input length."

- **Samples**: 50
- **Dataset**: `trec_coarse` split
- **Metric**:  
  - Numerical answers: `score = 0.75^|y - ŷ|`  
  - Others: exact match

### Three experimental conditions (per dataset)

| Condition    | Description |
|-------------|-------------|
| **Plain LLM** | Direct single LLM call — no RLM, no REPL, no recursion |
| **RLM depth=1** | RLM framework with max_depth=1 (REPL + code, but no recursive sub-calls) |
| **RLM depth=2** | RLM framework with max_depth=2 (allows one level of recursive sub-LLM calls) |

All three conditions use the **same underlying model** (DeepSeek) and the
**same data / evaluation**, so differences are purely from the RLM framework.

## Quickstart

### 1. Install dependencies

```bash
# RLM (from repo root)
cd rlm && pip install -e .

# Experiment dependencies
cd ..
pip install datasets python-dotenv openai
```

### 2. Configure API keys

```bash
cd experiments
cp .env.template .env
# Edit .env and fill DEEPSEEK_API_KEY or OPENAI_API_KEY
```

### 3. Run experiments

```bash
# RULER — 3 conditions
python run_ruler_plain.py      # Plain LLM (no RLM)
python run_ruler_baseline.py   # RLM depth=1
python run_ruler_depth2.py     # RLM depth=2

# OOLONG — 3 conditions
python run_oolong_plain.py     # Plain LLM (no RLM)
python run_oolong_depth1.py    # RLM depth=1
python run_oolong_depth2.py    # RLM depth=2

# Compare results
python compare_results.py
```

### 4. Outputs

Results are written to the `results/` folder:
- `ruler_plain_llm_results.json`
- `ruler_depth1_results.json`
- `ruler_depth2_results.json`
- `oolong_plain_llm_results.json`
- `oolong_depth1_results.json`
- `oolong_depth2_results.json`
- `comparison_summary.json`

## File overview

### Main experiment scripts

| File | Description |
|------|-------------|
| `run_ruler_plain_llm.py` | RULER — plain LLM driver (no RLM) |
| `run_ruler_experiment.py` | RULER — RLM experiment driver |
| `run_oolong_plain_llm.py` | OOLONG — plain LLM driver (no RLM) |
| `run_oolong_experiment.py` | OOLONG — RLM experiment driver |
| `compare_results.py` | Aggregate and compare all 6 runs |

### Convenience entry points

| File | Description |
|------|-------------|
| `run_ruler_plain.py` | RULER Plain LLM (50 samples) |
| `run_ruler_baseline.py` | RULER RLM depth=1 |
| `run_ruler_depth2.py` | RULER RLM depth=2 |
| `run_oolong_plain.py` | OOLONG Plain LLM (trec_coarse, 50 samples) |
| `run_oolong_depth1.py` | OOLONG RLM depth=1 |
| `run_oolong_depth2.py` | OOLONG RLM depth=2 |

## Metrics

### RULER S-NIAH
- **Accuracy**: exact match of the target needle.

### OOLONG
- **Numerical**: `score = 0.75^|y - ŷ|` (partial credit)
- **Others**: exact match (1 or 0)

## Cost notes

Using DeepSeek API (roughly ¥1 / 1M tokens as a ballpark):
- 50 samples × 6 experiments ≈ 300 calls
- Plain LLM experiments are cheapest (single call per sample)
- RLM depth=2 experiments are most expensive (recursive sub-calls)

## Practical notes

1. **Rate limits**: if you hit API rate limits, consider adding `time.sleep()` between calls or reducing concurrent runs.  
2. **Network**: OOLONG is loaded from HuggingFace; make sure your machine can access `datasets` downloads.  
3. **trec_coarse**: the OOLONG experiments here always restrict to the `trec_coarse` split as in the RLM paper.
