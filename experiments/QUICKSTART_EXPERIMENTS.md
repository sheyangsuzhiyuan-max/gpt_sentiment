# 实验系统快速指南

## 🎯 目标

通过系统性实验，为简历积累以下要点：
- ✅ Fine-tuned BERT/Transformers
- ✅ Hyperparameter Tuning
- ✅ Transfer Learning (Layer Freezing)
- ✅ Model Optimization
- ✅ Experiment Management

## 🚀 快速开始（3步）

### 步骤1: 运行单个实验

```bash
# 运行基线实验
python experiments/run_experiment.py --config experiments/configs/baseline.yaml

# 运行层冻结实验
python experiments/run_experiment.py --config experiments/configs/freeze_layers.yaml
```

### 步骤2: 批量运行所有实验

```bash
# 一键运行所有配置（推荐在服务器上运行）
bash experiments/run_all.sh
```

### 步骤3: 查看对比结果

```bash
# 生成对比图和报告
python experiments/compare_experiments.py

# 生成简历内容
python experiments/generate_resume_points.py
```

## 📁 实验结果位置

运行后会生成以下内容：

```
experiments/
├── results/
│   ├── exp_001_baseline/
│   │   ├── config.yaml              # 实验配置
│   │   ├── metrics.json             # 评估指标
│   │   ├── confusion_matrix.png     # 混淆矩阵
│   │   ├── classification_report.txt
│   │   └── training_log.txt
│   ├── exp_002_freeze_4_layers/
│   └── ...
├── experiment_tracker.csv           # 所有实验汇总表
├── experiment_comparison.png        # 对比可视化
├── EXPERIMENT_REPORT.md            # 实验报告
└── RESUME_CONTENT.md               # 简历内容
```

## 🎓 建议的实验流程

### 第一阶段：基础实验（必做）

1. **Baseline** - 建立性能基准
   ```bash
   python experiments/run_experiment.py --config experiments/configs/baseline.yaml
   ```

2. **Layer Freezing** - 测试迁移学习
   ```bash
   python experiments/run_experiment.py --config experiments/configs/freeze_layers.yaml
   ```

3. **Learning Rate** - 调优学习率
   ```bash
   python experiments/run_experiment.py --config experiments/configs/lower_lr.yaml
   python experiments/run_experiment.py --config experiments/configs/higher_lr.yaml
   ```

### 第二阶段：进阶实验（加分项）

4. **Heavy Freeze** - 只训练分类器
   ```bash
   python experiments/run_experiment.py --config experiments/configs/heavy_freeze.yaml
   ```

5. **自定义实验** - 创建新配置文件测试其他想法

## 📝 创建新实验

1. 复制现有配置文件：
   ```bash
   cp experiments/configs/baseline.yaml experiments/configs/my_experiment.yaml
   ```

2. 修改关键参数：
   ```yaml
   experiment:
     name: "my_experiment"
     description: "Testing XYZ hypothesis"

   training:
     learning_rate: 3.0e-5  # 修改这里
     batch_size: 64
   ```

3. 运行实验：
   ```bash
   python experiments/run_experiment.py --config experiments/configs/my_experiment.yaml
   ```

## 💡 实验想法清单

### 学习率相关
- [ ] 1e-5 (更保守)
- [ ] 2e-5 (baseline)
- [ ] 5e-5 (更激进)
- [ ] 学习率调度器对比（linear vs cosine）

### 层冻结相关
- [ ] 不冻结（全部训练）
- [ ] 冻结前2层
- [ ] 冻结前4层
- [ ] 冻结全部6层（只训练分类器）

### 正则化相关
- [ ] Weight Decay: 0, 0.001, 0.01, 0.1
- [ ] Dropout rate
- [ ] Gradient Clipping阈值

### 训练策略
- [ ] 不同batch size (16, 32, 64)
- [ ] Warmup steps (0, 500, 1000)
- [ ] 多epoch训练 (2-3 epochs)

### 高级技巧
- [ ] Label Smoothing
- [ ] Class Weights（处理不平衡）
- [ ] Mixup数据增强
- [ ] Gradual Unfreezing

## 📊 如何解读结果

### 查看实验汇总表

```bash
cat experiments/experiment_tracker.csv
```

关注以下指标：
- `test_acc`: 测试准确率（主要指标）
- `f1_macro`: F1分数（平衡指标）
- `train_acc - test_acc`: 过拟合程度
- `train_time_sec`: 训练效率

### 对比可视化

运行 `python experiments/compare_experiments.py` 后会生成：
- 准确率对比图
- F1-score趋势
- 学习率 vs 性能散点图
- 训练时间对比

## ✍️ 简历怎么写

运行 `python experiments/generate_resume_points.py` 会自动生成：

### 项目描述示例
```
情感分析系统 - BERT Fine-tuning
• 在 ChatGPT 评论数据集（220K 样本）上微调 DistilBERT 模型
• 系统性开展 10 组对照实验，调优学习率、层冻结等超参数
• 实施迁移学习策略（冻结4层），减少训练时间同时保持性能
• 最终模型测试准确率 93.2%，F1-score 0.92
• 部署为 Streamlit Web 应用，支持批量推理
```

### 面试可能的问题

1. **为什么选择 DistilBERT？**
   - 轻量级：参数量是BERT的60%
   - 高效：速度提升2倍
   - 性能：准确率仅降低3%
   - 适合部署和快速迭代

2. **层冻结的作用是什么？**
   - 底层学习通用语言特征
   - 冻结保留预训练知识
   - 只微调顶层适应特定任务
   - 减少训练时间和过拟合风险

3. **如何选择最佳超参数？**
   - 对比了X种学习率配置
   - 使用网格搜索/贝叶斯优化
   - 根据验证集性能选择
   - 考虑训练效率和性能平衡

4. **如何评估模型好坏？**
   - 准确率：整体性能
   - F1-score：平衡精确率和召回率
   - 混淆矩阵：各类别表现
   - 关注neutral类（最难分类）

## 🎯 时间规划

### 单个实验耗时
- 数据加载：~30秒
- 训练（1 epoch）：~10-20分钟（取决于设备）
- 评估和保存：~2分钟

### 建议安排
- **Day 1**: 运行基础实验（baseline + 2-3个变体）
- **Day 2**: 运行进阶实验，对比结果
- **Day 3**: 分析结果，准备简历和面试内容

## 🔧 服务器运行建议

如果在服务器上运行：

```bash
# 1. 使用 screen 或 tmux 保持会话
screen -S bert_experiments

# 2. 批量运行（后台）
nohup bash experiments/run_all.sh > experiments/run_all.log 2>&1 &

# 3. 监控进度
tail -f experiments/run_all.log

# 4. 下载结果
scp -r username@server:path/experiments/results ./
scp username@server:path/experiments/experiment_tracker.csv ./
```

## 📌 注意事项

1. **只保存Top-3模型**
   - 其他实验设置 `save_model: false`
   - 节省磁盘空间

2. **及时对比结果**
   - 每运行2-3个实验就对比一次
   - 及时调整方向

3. **记录实验笔记**
   - 在配置文件的`notes`字段记录想法
   - 有助于面试时回忆

4. **备份实验结果**
   - `experiment_tracker.csv` 是关键
   - 定期备份到云端

## 🎉 预期成果

完成实验系统后，你将拥有：

✅ **可量化的项目成果**
- X个实验配置
- 准确率提升Y%
- 完整的实验报告

✅ **简历亮点**
- Fine-tuning经验
- 超参数调优经验
- 实验管理能力

✅ **面试谈资**
- 具体的技术细节
- 实验对比数据
- 问题解决经验

---

**开始你的实验之旅吧！🚀**
