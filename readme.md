# ChatGPT 情感分析项目

基于 DistilBERT 的用户评论情感分析系统 - **生产级优化版**

> **CA6001 课程项目** | 完整的机器学习工作流 + 实验管理系统

## 📋 项目概述

端到端的情感分析系统，用于分类 ChatGPT 评论（正面/中性/负面）。

### 核心特点
- ✨ **统一配置管理** - `config.py` 集中管理所有超参数
- 🛠️ **模块化工具函数** - `utils.py` 提供可复用函数
- 🔄 **完全可复现** - 随机种子控制
- 🧪 **实验管理系统** - 系统性对比不同配置
- 🌐 **生产级 Web 应用** - 支持单条和批量分析

### 性能指标

| 模型 | 准确率 | F1-Score |
|------|--------|----------|
| Baseline (TF-IDF + LR) | ~75% | ~0.74 |
| RNN (LSTM) | ~90% | ~0.89 |
| **DistilBERT** | **93%** | **0.92** |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
conda create -n gpt_senti python=3.10
conda activate gpt_senti

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行数据预处理

```bash
cd notebooks
jupyter notebook 01_EDA_Preprocess.ipynb
```

### 3. 训练模型（可选）

```bash
# 方式1: 使用 notebook
jupyter notebook 04_BERT_Finetune.ipynb

# 方式2: 使用实验系统（推荐）
python experiments/run_experiment.py --config experiments/configs/baseline.yaml
```

### 4. 启动 Web 应用

```bash
streamlit run app.py
```

---

## 📁 项目结构

```
assignment_gpt_sentiment/
│
├── README.md                  # 项目文档（本文件）
├── config.py                  # 统一配置
├── utils.py                   # 工具函数
├── app.py                     # Web 应用
├── requirements.txt           # 依赖清单
├── .gitignore                # Git 配置
│
├── data/                     # 数据目录
│   ├── raw_data.csv
│   └── processed_data.csv
│
├── model_save/              # 模型保存
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
│
├── notebooks/               # Jupyter Notebooks
│   ├── 01_EDA_Preprocess.ipynb
│   ├── 02_Baseline_Model.ipynb
│   ├── 03_RNN_Model.ipynb
│   ├── 04_BERT_Finetune.ipynb
│   └── 05_evaluation.ipynb
│
├── experiments/             # 实验管理系统 ⭐
│   ├── configs/            # 实验配置
│   ├── results/            # 实验结果
│   ├── run_experiment.py   # 运行实验
│   └── compare_experiments.py
│
├── logs/                   # 日志文件
├── scripts/                # 工具脚本
└── docs/                   # 文档
    ├── QUICKSTART.md
    └── OPTIMIZATION_SUMMARY.md
```

---

## 🧪 实验管理系统

### 运行实验

```bash
# 单个实验
python experiments/run_experiment.py --config experiments/configs/baseline.yaml

# 批量运行所有实验
bash experiments/run_all.sh
```

### 对比结果

```bash
# 生成对比图和报告
python experiments/compare_experiments.py

# 生成简历内容
python experiments/generate_resume_points.py
```

### 实验配置

已预设 5 种配置：
- `baseline.yaml` - 默认配置
- `lower_lr.yaml` - 低学习率 (1e-5)
- `higher_lr.yaml` - 高学习率 (5e-5)
- `freeze_layers.yaml` - 冻结 4 层
- `heavy_freeze.yaml` - 只训练分类器

**详细说明**: 查看 [experiments/README.md](experiments/README.md)

---

## 📊 Web 应用功能

### 单条分析
1. 输入用户评论
2. 获取情感分类和置信度
3. 查看关键词和建议

### 批量分析
1. 上传包含 `text` 列的 CSV 文件
2. 批量推理
3. 下载结果

---

## 🔧 配置说明

所有配置在 [config.py](config.py) 中：

```python
RANDOM_SEED = 42              # 随机种子
MAX_SEQ_LENGTH = 128          # BERT 序列长度
BERT_BATCH_SIZE = 32          # 批次大小
BERT_LEARNING_RATE = 2e-5     # 学习率
TEST_SIZE = 0.2               # 测试集比例
```

修改配置后，所有脚本自动生效。

---

## 📝 简历内容

完成实验后可以写：

```
情感分析系统 - BERT Fine-tuning
• 在22万+用户评论上微调DistilBERT，测试准确率93%
• 系统性开展10组对照实验，调优学习率、层冻结等超参数
• 实施迁移学习策略（冻结transformer层），优化训练效率
• 建立自动化实验追踪系统，管理多维度评估指标
• 部署为Streamlit Web应用，支持批量推理
```

**技能标签**: Fine-tuning BERT, PyTorch, Transformers, Hyperparameter Tuning, Transfer Learning, NLP

---

## 🐛 常见问题

### Q: 模型加载失败？
A: 确保已运行 `04_BERT_Finetune.ipynb` 并保存模型到 `model_save/`

### Q: GPU/MPS 不可用？
A: 代码会自动回退到 CPU
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Q: 内存不足？
A: 在 `config.py` 中减小 `BERT_BATCH_SIZE`

### Q: 找不到数据文件？
A: 确保 `raw_data.csv` 在 `data/` 目录

### Q: ImportError: config
A: 确保在项目根目录运行，或在 notebook 中添加：
```python
import sys
sys.path.append('..')
```

---

## 📚 文档导航

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 主文档（本文件） |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 5分钟快速上手 |
| [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md) | 优化详情 |
| [experiments/README.md](experiments/README.md) | 实验系统说明 |
| [experiments/QUICKSTART_EXPERIMENTS.md](experiments/QUICKSTART_EXPERIMENTS.md) | 实验快速指南 |

---

## 🛠️ 工具脚本

```bash
# 测试环境
python scripts/test_environment.py

# 优化 notebooks（可选）
python scripts/optimize_notebooks.py
```

---

## 🎯 本次优化亮点

### 代码质量
- ✅ 统一配置文件
- ✅ 可复用工具函数
- ✅ 完善错误处理
- ✅ 代码注释和文档

### 实验管理
- ✅ 自动追踪所有实验
- ✅ 规范化结果保存
- ✅ 可视化对比
- ✅ 简历内容生成

### 可复现性
- ✅ 随机种子控制
- ✅ 日志记录
- ✅ 配置版本管理

### 生产部署
- ✅ Web 应用
- ✅ 批量推理
- ✅ 错误处理
- ✅ 性能优化

**详细改进**: 查看 [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md)

---

## 🔬 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.0+ |
| Transformer | Hugging Face Transformers |
| Web 框架 | Streamlit |
| 数据处理 | Pandas, NumPy |
| 可视化 | Matplotlib, Seaborn, Altair |
| 传统ML | scikit-learn |

---

## 📈 性能基准

测试集（43,858 样本）：

```
              precision    recall  f1-score   support
         bad       0.97      0.95      0.96     21,504
     neutral       0.86      0.89      0.87     11,075
        good       0.93      0.93      0.93     11,279

    accuracy                           0.93     43,858
   macro avg       0.92      0.92      0.92     43,858
weighted avg       0.93      0.93      0.93     43,858
```

---

## 👥 贡献者

- **课程**: CA6001 - AI Product Management
- **优化日期**: 2025-12-14

---

## 📄 许可证

本项目仅用于学术研究和教育目的。

---

## 🙏 致谢

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)

---

**快速开始**: 查看 [docs/QUICKSTART.md](docs/QUICKSTART.md)

**实验系统**: 查看 [experiments/QUICKSTART_EXPERIMENTS.md](experiments/QUICKSTART_EXPERIMENTS.md)
