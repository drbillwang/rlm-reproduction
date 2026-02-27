#!/bin/bash
# 快速运行实验脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

# 激活虚拟环境
if [ -d "$PROJECT_DIR/.venv" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
    echo "已激活虚拟环境"
else
    echo "警告: 虚拟环境不存在，请先运行: python3 -m venv .venv && source .venv/bin/activate && pip install -e ./rlm datasets rich"
    exit 1
fi

echo ""
echo "=========================================="
echo "RLM 实验运行器"
echo "=========================================="
echo ""
echo "可用模型:"
echo "  glm      - GLM 4 Plus (智谱)"
echo "  minimax  - MiniMax M2.5 (硅基流动)"
echo "  kimi     - Kimi K2.5 (硅基流动)"
echo "  deepseek - DeepSeek V3.2 (硅基流动)"
echo ""
echo "数据集: ruler, oolong"
echo "深度: 1, 2, 3"
echo ""

# 默认参数
MODELS=${1:-"deepseek"}
DATASETS=${2:-"ruler oolong"}
DEPTHS=${3:-"1 2 3"}
SAMPLES=${4:-50}

echo "运行配置:"
echo "  模型: $MODELS"
echo "  数据集: $DATASETS"
echo "  深度: $DEPTHS"
echo "  样本数: $SAMPLES"
echo ""

# 运行实验
python batch_run_experiments.py \
    --models $MODELS \
    --datasets $DATASETS \
    --depths $DEPTHS \
    --samples $SAMPLES

echo ""
echo "实验完成! 结果保存在 results/ 目录"
