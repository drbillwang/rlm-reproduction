"""
Compare Results Across All Experiments

This script compares results from three conditions per dataset:
- Plain LLM (no RLM)
- RLM depth=1
- RLM depth=2

For two datasets: RULER S-NIAH, OOLONG (trec_coarse).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


EXPERIMENT_CONFIGS = [
    {"key": "ruler_plain_llm", "dataset": "RULER",  "method": "Plain LLM"},
    {"key": "ruler_depth1",    "dataset": "RULER",  "method": "RLM depth=1"},
    {"key": "ruler_depth2",    "dataset": "RULER",  "method": "RLM depth=2"},
    {"key": "oolong_plain_llm","dataset": "OOLONG", "method": "Plain LLM"},
    {"key": "oolong_depth1",   "dataset": "OOLONG", "method": "RLM depth=1"},
    {"key": "oolong_depth2",   "dataset": "OOLONG", "method": "RLM depth=2"},
]


def load_results(experiment_name: str) -> Optional[Dict]:
    """Load results from JSON file."""
    path = Path(f"results/{experiment_name}_results.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compare_experiments():
    """Compare all experiments and generate summary."""

    print("=" * 70)
    print("EXPERIMENT RESULTS COMPARISON")
    print("Plain LLM  vs  RLM depth=1  vs  RLM depth=2")
    print("=" * 70)

    all_data: list[dict] = []

    for cfg in EXPERIMENT_CONFIGS:
        data = load_results(cfg["key"])
        if data:
            row = {
                "key": cfg["key"],
                "dataset": cfg["dataset"],
                "method": cfg["method"],
                "accuracy": data.get("accuracy", 0),
                "avg_score": data.get("avg_score", data.get("accuracy", 0)),
                "avg_time": data.get("avg_execution_time", 0),
                "samples": data.get("num_samples", 0),
                "correct": data.get("correct_count", 0),
                "total_tokens": data.get("total_tokens", 0),
            }
            all_data.append(row)
            print(f"  [{cfg['dataset']}] {cfg['method']}: "
                  f"Accuracy={row['accuracy']:.1f}%, "
                  f"Time={row['avg_time']:.2f}s")
        else:
            print(f"  [{cfg['dataset']}] {cfg['method']}: NOT FOUND")

    # --- summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print("\n| Dataset | Method       | Accuracy | Avg Score | Avg Time | Samples |")
    print("|---------|-------------|----------|-----------|----------|---------|")

    for d in all_data:
        print(f"| {d['dataset']:<7} | {d['method']:<11} | "
              f"{d['accuracy']:>6.1f}%  | {d['avg_score']:>7.1f}%  | "
              f"{d['avg_time']:>6.2f}s  | {d['samples']:>7} |")

    # --- improvement analysis ---
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS (vs Plain LLM baseline)")
    print("=" * 70)

    for ds in ["RULER", "OOLONG"]:
        ds_rows = [d for d in all_data if d["dataset"] == ds]
        plain = next((d for d in ds_rows if d["method"] == "Plain LLM"), None)
        if not plain:
            continue
        for d in ds_rows:
            if d["method"] == "Plain LLM":
                continue
            delta = d["accuracy"] - plain["accuracy"]
            print(f"{ds} {d['method']} vs Plain LLM: {delta:+.1f}%")

    # --- save ---
    ruler_rows = [d for d in all_data if d["dataset"] == "RULER"]
    oolong_rows = [d for d in all_data if d["dataset"] == "OOLONG"]

    comparison = {
        "ruler": ruler_rows,
        "oolong": oolong_rows,
        "summary": {
            "total_experiments": len(all_data),
            "model": "see individual result files",
        },
    }

    output_path = Path("results/comparison_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to {output_path}")

    # --- markdown for report ---
    print("\n" + "=" * 70)
    print("MARKDOWN TABLE (for report)")
    print("=" * 70)

    print("""
## Experiment Results

### RULER S-NIAH (Needle in a Haystack)

| Method       | Accuracy | Avg Score | Avg Time |
|-------------|----------|-----------|----------|""")
    for d in ruler_rows:
        print(f"| {d['method']:<11} | {d['accuracy']:.1f}% | "
              f"{d['avg_score']:.1f}% | {d['avg_time']:.2f}s |")

    print("""
### OOLONG (Long-context Reasoning, trec_coarse)

| Method       | Accuracy | Avg Score | Avg Time |
|-------------|----------|-----------|----------|""")
    for d in oolong_rows:
        print(f"| {d['method']:<11} | {d['accuracy']:.1f}% | "
              f"{d['avg_score']:.1f}% | {d['avg_time']:.2f}s |")

    return comparison


if __name__ == "__main__":
    compare_experiments()
