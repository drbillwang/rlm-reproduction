"""
RULER S-NIAH Baseline: max_depth=1

Paper: "Following the single needle-in-the-haystack task in RULER, 
we consider a set of 50 single tasks that require finding a specific 
phrase or number in a large set of unrelated text."
"""

import sys
sys.path.insert(0, '.')

from run_ruler_experiment import run_ruler_rlm_experiment

if __name__ == "__main__":
    run_ruler_rlm_experiment(
        max_depth=1,
        num_samples=50,  # Paper uses 50 samples
        max_seq_length=4096,
        output_name="ruler_depth1"
    )
