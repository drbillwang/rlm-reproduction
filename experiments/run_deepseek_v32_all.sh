#!/usr/bin/env bash

set -euo pipefail

# Move to the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[run_deepseek_v32_all] Activating virtual environment..."
source ../venv/bin/activate

mkdir -p logs

echo "[run_deepseek_v32_all] Starting RULER experiments with DeepSeek V3.2..."

echo "[run_deepseek_v32_all] RULER depth=1"
python3 run_ruler_baseline.py > logs/ruler_depth1_deepseek.log 2>&1

echo "[run_deepseek_v32_all] RULER depth=2"
python3 run_ruler_depth2.py > logs/ruler_depth2_deepseek.log 2>&1

echo "[run_deepseek_v32_all] RULER depth=3"
python3 run_ruler_depth3.py > logs/ruler_depth3_deepseek.log 2>&1

echo "[run_deepseek_v32_all] Starting OOLONG experiments with DeepSeek V3.2..."

echo "[run_deepseek_v32_all] OOLONG depth=1"
python3 run_oolong_depth1.py > logs/oolong_depth1_deepseek.log 2>&1

echo "[run_deepseek_v32_all] OOLONG depth=2"
python3 run_oolong_depth2.py > logs/oolong_depth2_deepseek.log 2>&1

echo "[run_deepseek_v32_all] OOLONG depth=3"
python3 run_oolong_depth3.py > logs/oolong_depth3_deepseek.log 2>&1

echo "[run_deepseek_v32_all] All 6 DeepSeek V3.2 experiments completed. Running compare_results.py..."
python3 compare_results.py > logs/compare_results_deepseek.log 2>&1 || echo "[run_deepseek_v32_all] compare_results.py failed (see logs)."

echo "[run_deepseek_v32_all] Done."

