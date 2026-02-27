"""
RULER S-NIAH: Plain LLM (no RLM) baseline
"""

import sys
sys.path.insert(0, '.')

from run_ruler_plain_llm import run_ruler_plain_experiment

if __name__ == "__main__":
    run_ruler_plain_experiment(
        num_samples=20,
        max_seq_length=4096,
        output_name="ruler_plain_llm"
    )
