# RLM 实验脚本

本文件夹包含复现 RLM 论文实验的脚本，使用 RULER 和 OOLONG 两个基准数据集。

## 实验设计（按论文要求）

### S-NIAH (RULER)
> "Following the single needle-in-the-haystack task in RULER, we consider a set of 50 single tasks that require finding a specific phrase or number in a large set of unrelated text. Here, the information being sought scales as O(1) with respect to input length."

- **样本数**: 50 samples
- **任务**: Needle in a Haystack 检索
- **评估**: 精确匹配

### OOLONG
> "We focus specifically on the trec_coarse split, a set of 50 tasks over a dataset of questions with semantic labels. Each task requires using nearly all entries of the dataset, and therefore scales linearly in processing complexity relative to the input length."

- **样本数**: 50 tasks
- **数据集**: trec_coarse split
- **评估**: 数值答案 `score = 0.75^|y - ŷ|`，其他精确匹配

### 修改变量
- `max_depth = 1` (baseline)
- `max_depth = 2` (modification 1)
- `max_depth = 3` (modification 2)

## 快速开始

### 1. 安装依赖

```bash
# RLM 依赖
cd rlm && pip install -e .

# 实验依赖
pip install datasets python-dotenv
```

### 2. 配置 API 密钥

```bash
cd experiments
cp .env.template .env
# 编辑 .env 文件，填入你的 DeepSeek API 密钥
```

### 3. 运行实验

```bash
# RULER 实验 (每个约 10-15 分钟)
python run_ruler_baseline.py   # depth=1 (baseline)
python run_ruler_depth2.py     # depth=2
python run_ruler_depth3.py     # depth=3

# OOLONG 实验 (每个约 10-15 分钟)
python run_oolong_depth1.py    # depth=1 (baseline)
python run_oolong_depth2.py    # depth=2
python run_oolong_depth3.py    # depth=3

# 对比结果
python compare_results.py
```

### 4. 查看结果

结果保存在 `results/` 文件夹：
- `ruler_depth1_results.json`
- `ruler_depth2_results.json`
- `ruler_depth3_results.json`
- `oolong_depth1_results.json`
- `oolong_depth2_results.json`
- `oolong_depth3_results.json`
- `comparison_summary.json`

## 文件说明

### 主要实验脚本

| 文件 | 说明 |
|------|------|
| `run_ruler_experiment.py` | RULER 实验主脚本 |
| `run_oolong_experiment.py` | OOLONG 实验主脚本 |
| `compare_results.py` | 结果对比与报告生成 |

### 快捷运行脚本

| 文件 | 说明 |
|------|------|
| `run_ruler_baseline.py` | RULER depth=1 (50 samples) |
| `run_ruler_depth2.py` | RULER depth=2 |
| `run_ruler_depth3.py` | RULER depth=3 |
| `run_oolong_depth1.py` | OOLONG depth=1 (trec_coarse, 50 samples) |
| `run_oolong_depth2.py` | OOLONG depth=2 |
| `run_oolong_depth3.py` | OOLONG depth=3 |

## 自定义实验

```bash
# 自定义参数运行 RULER
python run_ruler_experiment.py --depth 2 --samples 50 --length 8192

# 自定义参数运行 OOLONG
python run_oolong_experiment.py --depth 2 --samples 50 --dataset trec_coarse
```

## 评估指标

### RULER S-NIAH
- **Accuracy**: 精确匹配准确率（找到正确的 needle）

### OOLONG
- **Numerical**: `score = 0.75^|y - ŷ|`（部分正确）
- **Others**: 精确匹配

## 预计成本

使用 DeepSeek API（约 ¥1/百万 token）：
- 50 样本 × 6 实验 = 300 次 API 调用
- 预计成本：¥30-50

## 注意事项

1. **API 限流**: 如果遇到限流，可以在脚本中添加 `time.sleep()`
2. **网络问题**: OOLONG 需要从 HuggingFace 下载数据
3. **trec_coarse**: 论文指定使用 trec_coarse split
