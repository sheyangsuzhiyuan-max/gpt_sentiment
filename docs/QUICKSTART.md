# 快速开始指南

这是一个5分钟快速上手指南。

## 第一步：测试环境

```bash
# 运行环境测试
python test_environment.py
```

如果看到 `transformers` 未安装，运行：
```bash
pip install transformers
```

## 第二步：检查数据

确保 `data/raw_data.csv` 存在，包含以下列：
- `tweets`: 文本内容
- `labels`: 情感标签（bad/neutral/good）

## 第三步：运行数据预处理

```bash
cd notebooks
jupyter notebook 01_EDA_Preprocess.ipynb
```

按顺序执行所有 cell，将生成 `data/processed_data.csv`

## 第四步：训练模型（可选）

如果你想重新训练模型：

```bash
# 在 notebooks 目录下
jupyter notebook 04_BERT_Finetune.ipynb
```

**注意**：训练需要 10-30 分钟（取决于设备）

如果已有训练好的模型在 `model_save/`，跳过此步骤。

## 第五步：启动 Web 应用

```bash
# 返回项目根目录
cd ..

# 启动应用
streamlit run app.py
```

浏览器会自动打开应用界面。

## 使用 Web 应用

### 单条分析
1. 在文本框输入评论，如："I love ChatGPT!"
2. 点击"开始分析"
3. 查看结果：情感分类 + 置信度 + 建议

### 批量分析
1. 准备 CSV 文件（包含 `text` 列）
2. 切换到"批量分析"标签
3. 上传文件
4. 点击"开始批量分析"
5. 下载结果

## 常见问题

### Q: 模型加载失败？
A: 确保 `model_save/` 目录存在且包含以下文件：
- config.json
- pytorch_model.bin
- tokenizer_config.json
- vocab.txt

### Q: 如何修改配置？
A: 编辑 `config.py` 文件，所有脚本会自动使用新配置。

### Q: 内存不足？
A: 在 `config.py` 中减小 `BERT_BATCH_SIZE`：
```python
BERT_BATCH_SIZE = 16  # 或 8
```

## 项目结构速览

```
📂 项目根目录
├── config.py           # 配置文件（修改这里）
├── utils.py            # 工具函数
├── app.py              # Web 应用
├── test_environment.py # 环境测试
├── data/               # 数据目录
├── model_save/         # 模型目录
├── logs/               # 日志目录
└── notebooks/          # 分析脚本
```

## 下一步

- 📖 阅读完整文档：[README.md](README.md)
- 🔧 查看配置选项：[config.py](config.py)
- 🛠️ 学习工具函数：[utils.py](utils.py)
- 📊 探索数据分析：`notebooks/01_EDA_Preprocess.ipynb`

## 获取帮助

- 查看日志：`logs/` 目录
- 运行测试：`python test_environment.py`
- 查看 FAQ：README.md 的"常见问题"部分

---

**祝使用顺利！🚀**
