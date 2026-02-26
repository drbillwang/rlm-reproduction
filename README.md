# rlm-reproduction: Recursive Language Models Reproduction (FTEC5660)

This repository contains my individual project for FTEC5660, reproducing key
experiments from **“Recursive Language Models (RLM)”** and studying how the
recursion depth `max_depth` affects performance and cost on long‑context
benchmarks.

The goal is to:
- Reproduce the **S‑NIAH** and **OOLONG (trec_coarse)** results from the paper
  as faithfully as possible (same tasks, metrics, and evaluation code), and  
- Make one small but meaningful modification: sweep **`max_depth ∈ {1,2,3}`**
  while keeping everything else fixed, then analyze the impact on accuracy,
  latency, and token/cost usage.

---

## Repository structure

```text
rlm-reproduction/
├── rlm/          # Upstream RLM inference library (subdir of the original repo)
├── RULER/        # RULER benchmark code (for S‑NIAH data generation)
├── oolong/       # OOLONG benchmark code and helpers
└── experiments/  # My reproduction scripts
    ├── run_ruler_experiment.py   # S‑NIAH (RULER) with RLM + DeepSeek
    ├── run_ruler_baseline.py     # depth=1 shortcut
    ├── run_ruler_depth2.py       # depth=2 shortcut
    ├── run_ruler_depth3.py       # depth=3 shortcut
    ├── run_oolong_experiment.py  # OOLONG trec_coarse with RLM + DeepSeek
    ├── run_oolong_depth1.py      # depth=1 shortcut
    ├── run_oolong_depth2.py      # depth=2 shortcut
    ├── run_oolong_depth3.py      # depth=3 shortcut
    ├── compare_results.py        # Aggregate + compare all runs
    └── README.md                 # Detailed run instructions
```

Upstream projects:
- RLM: original implementation of Recursive Language Models  
- RULER: long‑context synthetic benchmark suite (for S‑NIAH)  
- OOLONG: long‑context reasoning benchmark (for OOLONG / trec_coarse)

I do **not** modify the core RLM algorithm; all my changes live under
`experiments/` and are wired on top of the upstream libraries.

---

## Target experiments (what I reproduce)

From the RLM paper, I focus on two benchmarks:

- **S‑NIAH (RULER)**  
  - “Single needle‑in‑the‑haystack” task from RULER (Hsieh et al., 2024)  
  - I use RULER’s official `niah_single_2` configuration:
    essay haystack + one numeric needle.  
  - I follow the paper and evaluate on **50 single tasks** at length 4k tokens.  
  - Metric: **accuracy** (percentage of correctly retrieved needles).

- **OOLONG (trec_coarse)**  
  - Long‑context reasoning/aggregation benchmark from OOLONG (Bertsch et al., 2025).  
  - I restrict to the **`trec_coarse`** split and randomly select **50 tasks**,
    as described in the RLM paper.  
  - Metric: same as the OOLONG paper and RLM paper:
    - Numerical answers: \(score(\hat{y}) = 0.75^{|y - \hat{y}|}\)  
    - Others: exact match.

For both benchmarks, I use the **official data generation / evaluation code**
from RULER and OOLONG wherever possible, and only change:

1. The **underlying LLM** (DeepSeek via OpenAI‑compatible API instead of GPT‑5/Qwen), and  
2. The **RLM recursion depth** (`max_depth`).

---

## Modification: max_depth sweep

The main controlled variable in this project is the **RLM recursion depth**:

- `max_depth = 1`: only the root RLM writes code in the REPL; no recursive
  `rlm_query` calls (falls back to plain `llm_query`).  
- `max_depth = 2`: the root RLM can spawn one additional child RLM per
  sub‑problem.  
- `max_depth = 3`: two levels of recursive sub‑RLMs are allowed.

All other settings (datasets, prompts, evaluation scripts, model, API keys)
are kept fixed across runs. For each depth and each benchmark I log:

- Accuracy and average score  
- Average execution time per task  
- Total input/output tokens and approximate total cost in USD  
- Per‑example response, score, tokens and cost

This allows direct comparison with the qualitative trends in the paper
(e.g. Figure 1, Table 1, Figure 3), while clearly isolating `max_depth` as
the main modification.

---

## Setup

### 1. Python environment

I assume Python 3.10+ and a virtual environment.

```bash
# From repo root
cd rlm
pip install -e .          # install RLM in editable mode

cd ..
pip install datasets python-dotenv
```

### 2. API keys (DeepSeek or OpenAI)

In `experiments/`:

```bash
cd experiments
cp .env.template .env
```

Edit `.env` and fill in **one** of:

- `DEEPSEEK_API_KEY` and optional `DEEPSEEK_BASE_URL`
- or `OPENAI_API_KEY`

You can also override the model via `MODEL_NAME` (default: `deepseek-chat`).

> NOTE: `.env` is **git‑ignored**. No secrets are committed to GitHub.

---

## Running the experiments

From `experiments/`:

```bash
cd experiments

# 1. RULER S‑NIAH (50 samples, depth = 1,2,3)
python run_ruler_baseline.py      # depth=1
python run_ruler_depth2.py        # depth=2
python run_ruler_depth3.py        # depth=3

# 2. OOLONG trec_coarse (50 tasks, depth = 1,2,3)
python run_oolong_depth1.py       # depth=1
python run_oolong_depth2.py       # depth=2
python run_oolong_depth3.py       # depth=3

# 3. Aggregate and compare all results
python compare_results.py
```

All raw results are saved under `results/` as JSON files, e.g.:

- `ruler_depth1_results.json`, `ruler_depth2_results.json`, `ruler_depth3_results.json`
- `oolong_depth1_results.json`, `oolong_depth2_results.json`, `oolong_depth3_results.json`
- `comparison_summary.json`

These files include both aggregate metrics and per‑example logs, so the plots
and tables in the write‑up can always be regenerated.

---

## Limitations and differences vs. the paper

- **Base model**: I use **DeepSeek** (via OpenAI‑compatible API) instead of
  GPT‑5 / Qwen, so absolute scores are **not directly comparable**. I focus
  on matching tasks and metrics, and comparing **relative trends**.
- **System prompt**: I use the default `RLM_SYSTEM_PROMPT` shipped with the
  open‑source `rlm` repo, which closely follows the GPT‑5 prompt in the
  appendix but is not a character‑for‑character copy.
- **Length / subsets**: I reproduce S‑NIAH at 4k and OOLONG trec_coarse with
  50 tasks; I do **not** run BrowseComp‑Plus, OOLONG‑Pairs, or CodeQA due to
  budget and time constraints.

These differences will be documented explicitly in the report; all other
choices (datasets, metrics, evaluation scripts) follow the paper and the
upstream repos as closely as possible.

---

## Academic honesty

This repository is for the FTEC5660 individual project. All reproduction
code written by me lives in `experiments/`. Upstream code from RLM, RULER
and OOLONG is used as‑is and properly cited in the report.

---

## References

- **Recursive Language Models (RLM)**  
  Paper: [Recursive Language Models](https://arxiv.org/abs/2512.24601)  
  Code: [`alexzhang13/rlm`](https://github.com/alexzhang13/rlm)


