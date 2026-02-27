"""
OOLONG Baseline: max_depth=1

Paper: "We focus specifically on the trec_coarse split, a set of 50 tasks 
over a dataset of questions with semantic labels. Each task requires using 
nearly all entries of the dataset, and therefore scales linearly in 
processing complexity relative to the input length."
"""

import sys
sys.path.insert(0, '.')

from run_oolong_experiment import run_oolong_experiment

if __name__ == "__main__":
    run_oolong_experiment(
        max_depth=1,
        num_samples=20,
        output_name="oolong_depth1",
        dataset_filter="trec_coarse"  # Paper focuses on trec_coarse split
    )
