# ChatGPT 情感分析项目 (Sentiment Analysis for ChatGPT Reviews)

基于 DistilBERT 的用户评论情感分析系统 - **优化版**

> **CA6001 课程项目** | 完整的机器学习工作流 + 生产级代码优化

## 📋 项目概述

本项目实现了一个端到端的情感分析系统，用于分析 ChatGPT 相关评论的情感倾向。

### 核心特点
- ✨ **统一配置管理** - 使用 `config.py` 集中管理所有超参数
- 🛠️ **模块化工具函数** - `utils.py` 提供可复用的工具函数
- 🔄 **完全可复现** - 随机种子控制确保结果一致
- 📝 **完善的日志记录** - 追踪所有训练过程
- 🔒 **健壮的错误处理** - 输入验证和异常处理
- 🌐 **生产级 Web 应用** - 支持单条和批量分析

### 性能指标

| 模型 | 准确率 | F1-Score (Macro) | 备注 |
|------|--------|------------------|------|
| Baseline (TF-IDF + LR) | ~75% | ~0.74 | 传统机器学习基线 |
| RNN (LSTM) | ~90% | ~0.89 | 简单深度学习模型 |
| **DistilBERT (Fine-tuned)** | **93%** | **0.92** | 最优模型 ✨ |

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- macOS (支持 Apple Silicon MPS 加速) / Linux / Windows
- 至少 8GB RAM（训练 BERT 建议 16GB+）

### 2. 安装依赖

```bash
# 克隆项目
cd assignment_gpt_sentiment

# 创建虚拟环境（推荐）
conda create -n gpt_senti python=3.10
conda activate gpt_senti

# 安装依赖
pip install pandas numpy matplotlib seaborn scikit-learn
pip install torch torchvision torchaudio
pip install transformers streamlit altair jupyter
```

### 3. 准备数据

将原始数据文件放入 `data/` 目录：

```
data/
├── raw_data.csv          # 原始数据（包含 tweets 和 labels 列）
└── processed_data.csv    # 运行 notebook 后自动生成
```

### 4. 运行 Notebooks

按顺序运行以下 notebooks：

```bash
cd notebooks

# 1. 数据探索与预处理（已优化）
jupyter notebook 01_EDA_Preprocess.ipynb

# 2-5. 其他notebooks按需运行
jupyter notebook 04_BERT_Finetune.ipynb  # 推荐：直接训练BERT
```

### 5. 启动 Web 应用

```bash
# 返回项目根目录
cd ..

# 启动应用
streamlit run app.py
```

应用将在浏览器中打开（默认 http://localhost:8501）

---

## 📁 项目结构

```
assignment_gpt_sentiment/
│
├── config.py                  # ⚙️ 统一配置文件（新增）
├── utils.py                   # 🛠️ 工具函数模块（新增）
├── app.py                     # 🌐 Streamlit Web 应用（已优化）
├── .gitignore                 # Git 忽略文件（新增）
├── README.md                  # 📖 项目文档（更新）
│
├── data/                      # 数据目录
│   ├── raw_data.csv          # 原始数据
│   └── processed_data.csv    # 处理后数据
│
├── model_save/               # 模型保存目录
│   ├── config.json           # BERT 配置
│   ├── pytorch_model.bin     # 模型权重
│   ├── tokenizer_config.json # Tokenizer 配置
│   ├── vocab.txt             # 词表
│   └── metadata.txt          # 元数据（新增）
│
├── logs/                     # 📝 日志目录（新增）
│   ├── eda_preprocess.log
│   ├── baseline_model.log
│   └── ...
│
└── notebooks/                # Jupyter Notebooks
    ├── 01_EDA_Preprocess.ipynb      # ✅ 已优化
    ├── 02_Baseline_Model.ipynb      # 可使用原版
    ├── 03_RNN_Model.ipynb           # 可使用原版
    ├── 04_BERT_Finetune.ipynb       # 可使用原版
    └── 05_evaluation.ipynb          # 可使用原版```

---

## 🔧 配置说明

所有超参数和路径配置都在 [config.py](config.py) 中管理：

```python
# 关键配置示例
RANDOM_SEED = 42              # 随机种子（确保可复现）
MAX_SEQ_LENGTH = 128          # BERT 最大序列长度
BERT_BATCH_SIZE = 32          # 批次大小
BERT_LEARNING_RATE = 2e-5     # 学习率
BERT_EPOCHS = 1               # 训练轮数
TEST_SIZE = 0.2               # 测试集比例
```

修改配置后，所有 notebooks 和应用都会自动使用新配置。

---

## 📊 使用示例

### 方式1：Web 应用（推荐）

1. 运行 `streamlit run app.py`
2. 在"单条分析"标签输入文本
3. 查看情感分类结果、置信度和建议

### 方式2：Python 代码

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import config

# 加载模型
tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_DIR)
model.eval()

# 预测
text = "I love ChatGPT but the API is too expensive"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)[0]
    pred = probs.argmax().item()

labels = list(config.LABEL_MAP.keys())
print(f"情感: {labels[pred]}")
print(f"置信度: {probs[pred]:.2%}")
```

### 方式3：批量分析

1. 准备 CSV 文件，包含 `text` 列
2. 在 Web 应用中切换到"批量分析"标签
3. 上传文件并点击"开始批量分析"
4. 下载分析结果

---

## 🎯 本次优化总结

### 代码质量改进
- ✅ 创建统一的 `config.py` 配置文件
- ✅ 创建 `utils.py` 工具函数模块
- ✅ 消除代码重复（Dataset 类、路径查找等）
- ✅ 添加完善的错误处理和输入验证
- ✅ 添加类型提示和文档字符串

### 可复现性改进
- ✅ 统一设置随机种子（Python、NumPy、PyTorch、MPS）
- ✅ 固定数据划分比例和参数
- ✅ 记录训练日志到文件
- ✅ 保存模型元数据

### 安全性改进
- ✅ 输入长度限制（防止内存溢出）
- ✅ 文件存在性检查
- ✅ 异常捕获和友好提示
- ✅ 模型加载失败时停止应用

### 性能优化
- ✅ 修复 05_evaluation.ipynb 的索引 bug
- ✅ BERT 训练时定期清理 MPS 缓存
- ✅ Streamlit 应用使用 `@st.cache_resource` 缓存模型
- ✅ 批量推理时显示进度条

### 功能增强
- ✅ Web 应用添加批量分析功能
- ✅ 添加详细的可视化（混淆矩阵、特征重要性）
- ✅ 改进关键词提取算法
- ✅ 添加使用说明标签页
- ✅ 支持 CSV 结果下载

### 文档完善
- ✅ 详细的 README 文档
- ✅ 代码注释和文档字符串
- ✅ 使用说明和示例
- ✅ 常见问题解答

---

## 🐛 常见问题

### Q1: 模型加载失败
**A:** 确保已运行 `04_BERT_Finetune.ipynb` 并成功保存模型到 `model_save/` 目录。

### Q2: GPU/MPS 不可用
**A:** 代码会自动回退到 CPU。检查 PyTorch 安装：
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Q3: 内存不足
**A:** 减小 batch_size：
```python
# 在 config.py 中修改
BERT_BATCH_SIZE = 16  # 或更小（8, 4）
```

### Q4: 找不到数据文件
**A:** `utils.find_data_file()` 会自动搜索多个路径。确保文件在以下位置之一：
- `data/raw_data.csv`
- `../data/raw_data.csv`（从 notebooks/ 运行时）

### Q5: ImportError: No module named 'config'
**A:** 确保在项目根目录运行，或在 notebook 中添加：
```python
import sys
sys.path.append('..')
```

### Q6: Streamlit 应用报错
**A:** 检查：
1. 是否安装了所有依赖
2. 是否有训练好的模型在 `model_save/`
3. 查看具体错误信息和日志文件

---

## 📈 性能基准

在测试集上的详细性能（~43,858 样本）：

### DistilBERT 模型（推荐）
```
              precision    recall  f1-score   support
         bad       0.97      0.95      0.96     21,504
     neutral       0.86      0.89      0.87     11,075
        good       0.93      0.93      0.93     11,279

    accuracy                           0.93     43,858
   macro avg       0.92      0.92      0.92     43,858
weighted avg       0.93      0.93      0.93     43,858
```

**关键发现**：
- 模型在识别"bad"情感时最准确（precision 0.97）
- "neutral"类别最具挑战性（precision 0.86）
- 整体性能优于基线模型 18%

---

## 🔬 技术栈

| 组件 | 技术 | 版本要求 |
|------|------|---------|
| 深度学习框架 | PyTorch | ≥ 2.0 |
| Transformer | Hugging Face Transformers | ≥ 4.30 |
| Web 框架 | Streamlit | ≥ 1.20 |
| 数据处理 | Pandas, NumPy | Latest |
| 可视化 | Matplotlib, Seaborn, Altair | Latest |
| 传统ML | scikit-learn | ≥ 1.0 |

---

## 🚧 未来改进方向

- [ ] 支持更大的 BERT 模型（BERT-base, RoBERTa）
- [ ] 添加模型集成（Ensemble）
- [ ] 实现增量学习
- [ ] 添加解释性分析（SHAP, LIME）
- [ ] 支持多语言情感分析
- [ ] 部署到云服务（Docker + AWS/Azure）
- [ ] 添加 REST API（FastAPI）
- [ ] 实时数据流处理
- [ ] A/B 测试框架

---

## 📚 参考资料

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

---

## 👥 贡献者

- **课程**: CA6001 - AI Product Management
- **时间**: 2025年
- **优化日期**: 2025-12-13

---

## 📄 许可证

本项目仅用于学术研究和教育目的。

---

## 🙏 致谢

感谢以下开源项目：
- [Hugging Face](https://huggingface.co/) - Transformers 库
- [Streamlit](https://streamlit.io/) - Web 框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**祝使用愉快！如果觉得有帮助，请给项目点个星 ⭐**

---

## 📞 支持

如有问题或建议：
- 查看日志文件：`logs/*.log`
- 检查配置：`config.py`
- 查阅文档：本 README 和代码注释

**Happy Coding! 🚀**
