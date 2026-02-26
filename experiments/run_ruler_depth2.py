"""
RULER S-NIAH: max_depth=2
"""

import sys
sys.path.insert(0, '.')

from run_ruler_experiment import run_ruler_rlm_experiment

if __name__ == "__main__":
    run_ruler_rlm_experiment(
        max_depth=2,
        num_samples=50,
        max_seq_length=4096,
        output_name="ruler_depth2"
    )
