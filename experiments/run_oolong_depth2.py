"""
OOLONG: max_depth=2
"""

import sys
sys.path.insert(0, '.')

from run_oolong_experiment import run_oolong_experiment

if __name__ == "__main__":
    run_oolong_experiment(
        max_depth=2,
        num_samples=20,
        output_name="oolong_depth2",
        dataset_filter="trec_coarse"
    )
