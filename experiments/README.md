# 实验管理系统

本目录用于管理所有模型训练实验，方便对比不同超参数配置的效果。

## 目录结构

```
experiments/
├── README.md                    # 本文件
├── experiment_tracker.csv       # 实验记录表（自动生成）
├── run_experiment.py           # 实验运行脚本
├── configs/                    # 实验配置文件
│   ├── baseline.yaml
│   ├── freeze_layers.yaml
│   ├── higher_lr.yaml
│   └── ...
└── results/                    # 实验结果
    ├── exp_001_baseline/
    │   ├── config.yaml         # 实验配置
    │   ├── metrics.json        # 评估指标
    │   ├── confusion_matrix.png
    │   ├── training_log.txt
    │   └── classification_report.txt
    ├── exp_002_freeze/
    └── ...
```

## 快速开始

### 1. 运行基线实验
```bash
python experiments/run_experiment.py --config experiments/configs/baseline.yaml
```

### 2. 运行冻结层实验
```bash
python experiments/run_experiment.py --config experiments/configs/freeze_layers.yaml
```

### 3. 查看实验对比
```bash
python experiments/compare_experiments.py
```

## 简历可写内容

通过本实验系统，你可以在简历中写：

### 技能点
- ✅ **Fine-tuning Transformers**: 在 20 万+样本数据集上微调 DistilBERT
- ✅ **Hyperparameter Tuning**: 系统性调优学习率、权重衰减等超参数
- ✅ **Transfer Learning**: 实现层冻结策略，对比不同迁移学习方法
- ✅ **Model Optimization**: 通过实验对比提升模型性能 X%
- ✅ **Experiment Management**: 建立规范的实验追踪和版本管理系统

### 项目描述示例
```
情感分析系统 - BERT Fine-tuning
• 在 ChatGPT 评论数据集（220K 样本）上微调 DistilBERT 模型
• 实施多种优化策略：学习率调优、层冻结、梯度裁剪等
• 建立实验管理系统，系统性对比 10+ 组超参数配置
• 最终模型在测试集达到 93% 准确率，F1-score 0.92
• 部署为 Streamlit Web 应用，支持批量推理
```

## 实验想法清单

### 基础实验（必做）
1. **Baseline** - 默认配置
2. **Learning Rate** - 调整学习率（1e-5, 2e-5, 5e-5）
3. **Freeze Layers** - 冻结不同层数（freeze 0-6 layers）

### 进阶实验（加分项）
4. **Batch Size** - 调整批次大小（16, 32, 64）
5. **Weight Decay** - 添加权重衰减（0.01, 0.001）
6. **Gradient Clipping** - 梯度裁剪
7. **Learning Rate Scheduler** - 学习率调度（Warmup + Decay）
8. **Dropout** - 调整 dropout 率
9. **Class Weights** - 处理类别不平衡
10. **Ensemble** - 多模型集成

## 实验记录表字段

| 字段 | 说明 |
|------|------|
| exp_id | 实验编号（exp_001, exp_002, ...） |
| name | 实验名称 |
| config | 配置文件路径 |
| learning_rate | 学习率 |
| batch_size | 批次大小 |
| freeze_layers | 冻结层数 |
| train_acc | 训练准确率 |
| test_acc | 测试准确率 |
| f1_score | F1 分数 |
| train_time | 训练时长（秒） |
| timestamp | 时间戳 |
| notes | 备注 |

## 最佳实践

1. **每次实验前**
   - 创建新的配置文件
   - 给实验起个有意义的名字

2. **实验过程中**
   - 自动保存所有输出到对应目录
   - 不覆盖已有实验结果

3. **实验后**
   - 记录实验心得到 notes
   - 对比关键指标

4. **模型保存策略**
   - 只保存 Top-3 最优模型
   - 其他实验只保存评估报告
