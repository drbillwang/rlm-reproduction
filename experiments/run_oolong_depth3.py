"""
OOLONG: max_depth=3
"""

import sys
sys.path.insert(0, '.')

from run_oolong_experiment import run_oolong_experiment

if __name__ == "__main__":
    run_oolong_experiment(
        max_depth=3,
        num_samples=50,
        output_name="oolong_depth3",
        dataset_filter="trec_coarse"
    )
