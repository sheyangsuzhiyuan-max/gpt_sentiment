# 项目导航 - 快速索引

## 🎯 你想做什么？

### 📖 了解项目
→ 查看 [README.md](README.md)

### ⚡ 快速上手
→ 查看 [docs/QUICKSTART.md](docs/QUICKSTART.md)

### 🧪 运行实验（简历必备）
→ 查看 [experiments/QUICKSTART_EXPERIMENTS.md](experiments/QUICKSTART_EXPERIMENTS.md)

步骤：
```bash
# 1. 批量运行实验（约2小时）
bash experiments/run_all.sh

# 2. 查看对比结果
python experiments/compare_experiments.py

# 3. 生成简历内容
python experiments/generate_resume_points.py
```

### 🌐 使用 Web 应用
```bash
streamlit run app.py
```

### 🔧 修改配置
→ 编辑 [config.py](config.py)

### 🛠️ 查看工具函数
→ 查看 [utils.py](utils.py)

### 📊 查看实验结果
→ 查看 `experiments/results/` 和 `experiments/experiment_tracker.csv`

### 📝 准备简历
→ 查看 `experiments/RESUME_CONTENT.md`（运行实验后生成）

### 🐛 遇到问题
→ 查看 [README.md - 常见问题](README.md#-常见问题)

---

## 📂 项目结构速览

```
根目录/
├── README.md              ← 从这里开始
├── config.py              ← 修改配置
├── utils.py               ← 工具函数
├── app.py                 ← Web 应用
├── requirements.txt       ← 依赖清单
│
├── notebooks/             ← Jupyter 分析
│   ├── 01_EDA_Preprocess.ipynb
│   ├── 02_Baseline_Model.ipynb
│   ├── 03_RNN_Model.ipynb
│   ├── 04_BERT_Finetune.ipynb
│   └── 05_evaluation.ipynb
│
├── experiments/           ← 实验系统（重点！）
│   ├── configs/          ← 实验配置
│   ├── results/          ← 实验结果
│   ├── run_experiment.py ← 运行实验
│   └── compare_experiments.py ← 对比实验
│
├── data/                  ← 数据文件
├── model_save/            ← 保存的模型
├── logs/                  ← 日志文件
├── scripts/               ← 工具脚本
└── docs/                  ← 文档
    ├── INDEX.md                    ← 文档索引
    ├── QUICKSTART.md              ← 快速上手
    └── OPTIMIZATION_SUMMARY.md    ← 优化详情
```

---

## 📚 文档导航

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 项目主文档 |
| [docs/INDEX.md](docs/INDEX.md) | 文档索引 |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 5分钟上手 |
| [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md) | 优化详情（35+项改进） |
| [experiments/README.md](experiments/README.md) | 实验系统说明 |
| [experiments/QUICKSTART_EXPERIMENTS.md](experiments/QUICKSTART_EXPERIMENTS.md) | 实验快速指南 ⭐ |
| [experiments/EXPERIMENT_SYSTEM_SUMMARY.md](experiments/EXPERIMENT_SYSTEM_SUMMARY.md) | 实验系统总结 |

---

## 🚀 常用命令

```bash
# 测试环境
python scripts/test_environment.py

# 运行单个实验
python experiments/run_experiment.py --config experiments/configs/baseline.yaml

# 批量运行实验
bash experiments/run_all.sh

# 对比实验结果
python experiments/compare_experiments.py

# 生成简历内容
python experiments/generate_resume_points.py

# 启动 Web 应用
streamlit run app.py

# 运行 Jupyter
cd notebooks && jupyter notebook
```

---

## 🎯 简历准备流程

1. **运行实验**（2小时）
   ```bash
   bash experiments/run_all.sh
   ```

2. **查看对比**（5分钟）
   ```bash
   python experiments/compare_experiments.py
   ```

3. **生成简历内容**（2分钟）
   ```bash
   python experiments/generate_resume_points.py
   ```

4. **复制到简历**
   - 打开 `experiments/RESUME_CONTENT.md`
   - 复制简历要点
   - 准备面试问题

---

**开始探索**: 打开 [README.md](README.md) 👈
