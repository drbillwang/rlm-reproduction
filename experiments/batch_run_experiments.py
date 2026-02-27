"""
Batch experiment runner script.

Supports multiple models and recursion depths.
Two datasets: RULER and OOLONG.
Depths: 1, 2.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Model configuration (for security, do not hard-code real API keys here)
# Provide real keys via environment variables, for example:
#   GLM_API_KEY, GLM_BASE_URL, GLM_MODEL_NAME
#   SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, MINIMAX_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME
MODELS = {
    "glm": {
        "api_key": os.getenv("GLM_API_KEY", ""),
        "base_url": os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "model_name": os.getenv("GLM_MODEL_NAME", "glm-4-plus"),
    },
    "minimax": {
        "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "model_name": os.getenv("MINIMAX_MODEL_NAME", "Pro/MiniMaxAI/MiniMax-M2.5"),
    },
    "kimi": {
        "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "model_name": os.getenv("KIMI_MODEL_NAME", "Pro/moonshotai/Kimi-K2.5"),
    },
    "deepseek": {
        "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "model_name": os.getenv("DEEPSEEK_MODEL_NAME", "Pro/deepseek-ai/DeepSeek-V3.2"),
    },
}

# Experiment configuration
DEPTHS = [1, 2]
SAMPLES = 50  # Number of samples per experiment


def run_single_experiment(model_key: str, dataset: str, depth: int, samples: int = 50):
    """Run a single experiment."""
    model_config = MODELS[model_key]

    # Set environment variables for the child process
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = model_config["api_key"]
    env["OPENAI_BASE_URL"] = model_config["base_url"]
    env["DEEPSEEK_API_KEY"] = model_config["api_key"]
    env["DEEPSEEK_BASE_URL"] = model_config["base_url"]
    env["MODEL_NAME"] = model_config["model_name"]

    # Experiment name
    exp_name = f"{model_key}_{dataset}_depth{depth}"

    print("=" * 70)
    print(f"Running: {exp_name}")
    print(f"Model: {model_config['model_name']}")
    print(f"Dataset: {dataset}, Depth: {depth}, Samples: {samples}")
    print("=" * 70)

    # Run the experiment
    script = Path(__file__).parent / f"run_{dataset}_experiment.py"

    cmd = [
        "python", str(script),
        "--depth", str(depth),
        "--samples", str(samples),
        "--name", exp_name,
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = datetime.now()

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=False,  # Show live output
        text=True,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if result.returncode == 0:
        print(f"\n✓ {exp_name} completed in {duration:.1f}s")
        return True
    else:
        print(f"\n✗ {exp_name} failed!")
        return False


def run_all_experiments(models=None, datasets=None, depths=None, samples=50):
    """Run all experiment combinations."""
    if models is None:
        models = list(MODELS.keys())
    if datasets is None:
        datasets = ["ruler", "oolong"]
    if depths is None:
        depths = [1, 2]

    total = len(models) * len(datasets) * len(depths)
    completed = 0
    failed = 0

    print("=" * 70)
    print("BATCH EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Depths: {depths}")
    print(f"Samples per experiment: {samples}")
    print(f"Total experiments: {total}")
    print("=" * 70)
    print()

    results_log = []

    for model_key in models:
        for dataset in datasets:
            for depth in depths:
                success = run_single_experiment(model_key, dataset, depth, samples)

                if success:
                    completed += 1
                    status = "SUCCESS"
                else:
                    failed += 1
                    status = "FAILED"

                results_log.append({
                    "model": model_key,
                    "dataset": dataset,
                    "depth": depth,
                    "status": status,
                })

                print()
                print(f"Progress: {completed + failed}/{total} ({completed} success, {failed} failed)")
                print()

    # Final summary
    print("=" * 70)
    print("EXPERIMENT BATCH COMPLETE")
    print("=" * 70)
    print(f"Total: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print()

    # Save run log
    log_path = Path("results/batch_run_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": models,
            "datasets": datasets,
            "depths": depths,
            "samples": samples,
        },
        "summary": {
            "total": total,
            "completed": completed,
            "failed": failed,
        },
        "results": results_log,
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"Run log saved to: {log_path}")

    return completed, failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RLM experiments in batch")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run (glm, minimax, kimi, deepseek)")
    parser.add_argument("--datasets", nargs="+", default=["ruler", "oolong"],
                        help="Datasets to run (ruler, oolong)")
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2],
                        help="Recursion depths")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples per experiment")

    args = parser.parse_args()

    run_all_experiments(
        models=args.models,
        datasets=args.datasets,
        depths=args.depths,
        samples=args.samples,
    )
