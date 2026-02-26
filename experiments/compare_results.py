"""
Compare Results Across All Experiments

This script compares results from:
- RULER S-NIAH: depth=1, depth=2, depth=3
- OOLONG: depth=1, depth=2, depth=3

Generates summary tables and visualizations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def load_results(experiment_name: str) -> Optional[Dict]:
    """Load results from JSON file."""
    path = Path(f"results/{experiment_name}_results.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compare_experiments(
    ruler_depths: List[int] = [1, 2, 3],
    oolong_depths: List[int] = [1, 2, 3]
):
    """
    Compare all experiments and generate summary.
    """
    
    print("=" * 70)
    print("EXPERIMENT RESULTS COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # Load RULER results
    print("\n📊 RULER S-NIAH Results:")
    print("-" * 50)
    ruler_data = []
    for depth in ruler_depths:
        exp_name = f"ruler_depth{depth}"
        data = load_results(exp_name)
        if data:
            results[exp_name] = data
            ruler_data.append({
                "depth": depth,
                "accuracy": data.get("accuracy", 0),
                "avg_score": data.get("avg_score", data.get("accuracy", 0)),
                "avg_time": data.get("avg_execution_time", 0),
                "samples": data.get("num_samples", 0),
                "correct": data.get("correct_count", 0)
            })
            print(f"  depth={depth}: Accuracy={data.get('accuracy', 0):.1f}%, "
                  f"Time={data.get('avg_execution_time', 0):.2f}s")
        else:
            print(f"  depth={depth}: NOT FOUND (run the experiment first)")
    
    # Load OOLONG results
    print("\n📊 OOLONG Results:")
    print("-" * 50)
    oolong_data = []
    for depth in oolong_depths:
        exp_name = f"oolong_depth{depth}"
        data = load_results(exp_name)
        if data:
            results[exp_name] = data
            oolong_data.append({
                "depth": depth,
                "accuracy": data.get("accuracy", 0),
                "avg_score": data.get("avg_score", data.get("accuracy", 0)),
                "avg_time": data.get("avg_execution_time", 0),
                "samples": data.get("num_samples", 0),
                "correct": data.get("correct_count", 0)
            })
            print(f"  depth={depth}: Accuracy={data.get('accuracy', 0):.1f}%, "
                  f"Time={data.get('avg_execution_time', 0):.2f}s")
        else:
            print(f"  depth={depth}: NOT FOUND (run the experiment first)")
    
    # Generate comparison table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    
    # Table header
    print("\n| Dataset | Depth | Accuracy | Avg Score | Avg Time | Samples |")
    print("|---------|-------|----------|-----------|----------|---------|")
    
    for d in ruler_data:
        print(f"| RULER   | {d['depth']}     | {d['accuracy']:.1f}%    | "
              f"{d['avg_score']:.1f}%     | {d['avg_time']:.2f}s    | {d['samples']}       |")
    
    for d in oolong_data:
        print(f"| OOLONG  | {d['depth']}     | {d['accuracy']:.1f}%    | "
              f"{d['avg_score']:.1f}%     | {d['avg_time']:.2f}s    | {d['samples']}       |")
    
    # Calculate improvement
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    if len(ruler_data) >= 2:
        baseline = ruler_data[0]["accuracy"]
        for i, d in enumerate(ruler_data[1:], 1):
            improvement = d["accuracy"] - baseline
            print(f"RULER depth={d['depth']} vs baseline: {improvement:+.1f}%")
    
    if len(oolong_data) >= 2:
        baseline = oolong_data[0]["accuracy"]
        for i, d in enumerate(oolong_data[1:], 1):
            improvement = d["accuracy"] - baseline
            print(f"OOLONG depth={d['depth']} vs baseline: {improvement:+.1f}%")
    
    # Save comparison results
    comparison = {
        "ruler": ruler_data,
        "oolong": oolong_data,
        "summary": {
            "total_experiments": len(results),
            "model": list(results.values())[0].get("model", "unknown") if results else "unknown"
        }
    }
    
    output_path = Path("results/comparison_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to {output_path}")
    
    # Generate markdown table for report
    print("\n" + "=" * 70)
    print("MARKDOWN TABLE (for report)")
    print("=" * 70)
    
    print("""
## Experiment Results

### RULER S-NIAH (Needle in a Haystack)

| max_depth | Accuracy | Avg Execution Time |
|-----------|----------|-------------------|""")
    
    for d in ruler_data:
        print(f"| {d['depth']} | {d['accuracy']:.1f}% | {d['avg_time']:.2f}s |")
    
    print("""
### OOLONG (Long-context Reasoning)

| max_depth | Accuracy | Avg Execution Time |
|-----------|----------|-------------------|""")
    
    for d in oolong_data:
        print(f"| {d['depth']} | {d['accuracy']:.1f}% | {d['avg_time']:.2f}s |")
    
    return comparison


def generate_report_section():
    """Generate a section for the report."""
    
    results = {
        "ruler": [load_results(f"ruler_depth{i}") for i in [1, 2, 3]],
        "oolong": [load_results(f"oolong_depth{i}") for i in [1, 2, 3]]
    }
    
    report = """
## Reproduction Results

### Setup
- Model: DeepSeek (via API)
- Samples per experiment: 30
- Context length: 4096 tokens (RULER)

### Key Findings

"""
    
    # Add findings based on results
    ruler_results = [r for r in results["ruler"] if r]
    oolong_results = [r for r in results["oolong"] if r]
    
    if ruler_results:
        report += f"""
**RULER S-NIAH:**
- Baseline (depth=1): {ruler_results[0].get('accuracy', 0):.1f}% accuracy
"""
        if len(ruler_results) > 1:
            for i, r in enumerate(ruler_results[1:], 2):
                improvement = r.get('accuracy', 0) - ruler_results[0].get('accuracy', 0)
                report += f"- Depth={i}: {r.get('accuracy', 0):.1f}% ({improvement:+.1f}% vs baseline)\n"
    
    if oolong_results:
        report += f"""
**OOLONG:**
- Baseline (depth=1): {oolong_results[0].get('accuracy', 0):.1f}% accuracy
"""
        if len(oolong_results) > 1:
            for i, r in enumerate(oolong_results[1:], 2):
                improvement = r.get('accuracy', 0) - oolong_results[0].get('accuracy', 0)
                report += f"- Depth={i}: {r.get('accuracy', 0):.1f}% ({improvement:+.1f}% vs baseline)\n"
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Generate report section")
    args = parser.parse_args()
    
    if args.report:
        print(generate_report_section())
    else:
        compare_experiments()
