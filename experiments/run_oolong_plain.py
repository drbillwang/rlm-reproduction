"""
OOLONG: Plain LLM (no RLM) baseline
"""

import sys
sys.path.insert(0, '.')

from run_oolong_plain_llm import run_oolong_plain_experiment

if __name__ == "__main__":
    run_oolong_plain_experiment(
        num_samples=20,
        output_name="oolong_plain_llm",
        dataset_filter="trec_coarse"
    )
