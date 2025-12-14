"""
实验运行脚本
自动化训练、评估、结果保存
"""
import os
import sys
import yaml
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append('..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from tqdm import tqdm
import time

import config as base_config
import utils


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, config_path):
        """
        初始化实验运行器

        Args:
            config_path: 实验配置文件路径
        """
        self.config_path = config_path
        self.exp_config = self.load_config()
        self.exp_id = self.generate_exp_id()
        self.exp_dir = Path('experiments/results') / self.exp_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        utils.set_seed(self.exp_config['data']['random_seed'])

        # 设置设备
        self.device = base_config.DEVICE

        # 初始化日志
        self.logger = utils.setup_logger(
            f'Experiment_{self.exp_id}',
            log_file=self.exp_dir / 'training_log.txt'
        )

        self.logger.info(f"=" * 60)
        self.logger.info(f"实验开始: {self.exp_config['experiment']['name']}")
        self.logger.info(f"实验ID: {self.exp_id}")
        self.logger.info(f"=" * 60)

    def load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def generate_exp_id(self):
        """生成实验ID"""
        # 读取现有实验记录
        tracker_file = Path('experiments/experiment_tracker.csv')
        if tracker_file.exists():
            df = pd.read_csv(tracker_file)
            if len(df) > 0:
                last_id = int(df['exp_id'].str.replace('exp_', '').max())
                new_id = last_id + 1
            else:
                new_id = 1
        else:
            new_id = 1

        return f"exp_{new_id:03d}_{self.exp_config['experiment']['name']}"

    def load_data(self):
        """加载数据"""
        self.logger.info("加载数据...")

        # 读取处理后的数据
        data_path = utils.find_data_file(base_config.PROCESSED_DATA_FILE)
        df = pd.read_csv(data_path)

        # 验证数据
        required_cols = [base_config.CLEANED_TEXT_COLUMN, base_config.LABEL_COLUMN]
        utils.validate_dataframe(df, required_cols)

        # 删除缺失值
        df = df.dropna(subset=required_cols)

        # 创建标签ID
        if 'label_id' not in df.columns:
            df['label_id'] = df[base_config.LABEL_COLUMN].map(base_config.LABEL_MAP)

        self.logger.info(f"数据大小: {len(df):,}")

        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            df[base_config.CLEANED_TEXT_COLUMN],
            df['label_id'],
            test_size=self.exp_config['data']['test_size'],
            random_state=self.exp_config['data']['random_seed'],
            stratify=df['label_id']
        )

        self.logger.info(f"训练集: {len(X_train):,}, 测试集: {len(X_test):,}")

        return X_train, X_test, y_train, y_test

    def create_dataloaders(self, X_train, X_test, y_train, y_test):
        """创建数据加载器"""
        self.logger.info("创建数据加载器...")

        # 加载tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(
            self.exp_config['model']['name']
        )

        # 创建dataset
        max_len = self.exp_config['data']['max_seq_length']
        train_dataset = utils.SentimentDataset(X_train, y_train, tokenizer, max_len)
        test_dataset = utils.SentimentDataset(X_test, y_test, tokenizer, max_len)

        # 创建dataloader
        batch_size = self.exp_config['training']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return tokenizer, train_loader, test_loader

    def create_model(self):
        """创建模型"""
        self.logger.info("创建模型...")

        model = DistilBertForSequenceClassification.from_pretrained(
            self.exp_config['model']['name'],
            num_labels=self.exp_config['model']['num_labels']
        )

        # 冻结层
        freeze_layers = self.exp_config['model']['freeze_layers']
        if freeze_layers > 0:
            self.logger.info(f"冻结前 {freeze_layers} 层...")

            # DistilBERT 结构: embeddings + transformer (6 layers) + pre_classifier + classifier
            # 冻结 embeddings
            for param in model.distilbert.embeddings.parameters():
                param.requires_grad = False

            # 冻结指定数量的transformer层
            for i in range(freeze_layers):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False

            # 统计可训练参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.logger.info(f"总参数: {total_params:,}")
            self.logger.info(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

        model.to(self.device)
        return model

    def create_optimizer_scheduler(self, model, train_loader):
        """创建优化器和学习率调度器"""
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.exp_config['training']['learning_rate'],
            betas=tuple(self.exp_config['optimizer']['betas']),
            eps=self.exp_config['optimizer']['eps'],
            weight_decay=self.exp_config['training']['weight_decay']
        )

        # 学习率调度器
        scheduler_type = self.exp_config['scheduler'].get('type')

        if scheduler_type is None:
            scheduler = None

        elif scheduler_type == 'linear':
            num_training_steps = len(train_loader) * self.exp_config['training']['num_epochs']
            warmup_steps = self.exp_config['training'].get('warmup_steps', 0)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
            self.logger.info(f"使用线性学习率调度: warmup={warmup_steps}, total={num_training_steps}")

        elif scheduler_type == 'cosine':
            num_training_steps = len(train_loader) * self.exp_config['training']['num_epochs']
            warmup_steps = self.exp_config['training'].get('warmup_steps', 0)
            num_cycles = self.exp_config['scheduler'].get('num_cycles', 0.5)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles
            )
            self.logger.info(f"使用余弦学习率调度: warmup={warmup_steps}, cycles={num_cycles}")

        else:
            scheduler = None
            self.logger.warning(f"未知的调度器类型: {scheduler_type}")

        return optimizer, scheduler

    def train_epoch(self, model, train_loader, optimizer, scheduler, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            max_grad_norm = self.exp_config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # 统计
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

            # 定期清理缓存
            if (batch_idx + 1) % base_config.CACHE_CLEAR_INTERVAL == 0:
                utils.clear_mps_cache()

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def evaluate(self, model, test_loader):
        """评估模型"""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, preds = torch.max(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_labels), np.array(all_preds)

    def save_results(self, y_true, y_pred, metrics, train_time):
        """保存实验结果"""
        self.logger.info("保存实验结果...")

        # 保存配置
        config_file = self.exp_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.exp_config, f)

        # 保存指标
        metrics_file = self.exp_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # 保存分类报告
        if self.exp_config['output']['save_classification_report']:
            report = classification_report(
                y_true, y_pred,
                target_names=list(base_config.LABEL_MAP.keys()),
                digits=4
            )

            report_file = self.exp_dir / 'classification_report.txt'
            with open(report_file, 'w') as f:
                f.write(report)

            self.logger.info("\n" + report)

        # 保存混淆矩阵
        if self.exp_config['output']['save_confusion_matrix']:
            self.save_confusion_matrix(y_true, y_pred)

        # 更新实验追踪表
        self.update_tracker(metrics, train_time)

    def save_confusion_matrix(self, y_true, y_pred):
        """保存混淆矩阵图"""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 绝对数量
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(base_config.LABEL_MAP.keys()),
            yticklabels=list(base_config.LABEL_MAP.keys()),
            ax=axes[0]
        )
        axes[0].set_title(f'{self.exp_id} - Confusion Matrix (Count)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # 百分比
        sns.heatmap(
            cm_percent, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=list(base_config.LABEL_MAP.keys()),
            yticklabels=list(base_config.LABEL_MAP.keys()),
            ax=axes[1]
        )
        axes[1].set_title(f'{self.exp_id} - Confusion Matrix (%)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(self.exp_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"混淆矩阵已保存")

    def update_tracker(self, metrics, train_time):
        """更新实验追踪表"""
        tracker_file = Path('experiments/experiment_tracker.csv')

        # 准备记录
        record = {
            'exp_id': self.exp_id,
            'name': self.exp_config['experiment']['name'],
            'description': self.exp_config['experiment']['description'],
            'config_file': str(self.config_path),
            'learning_rate': self.exp_config['training']['learning_rate'],
            'batch_size': self.exp_config['training']['batch_size'],
            'freeze_layers': self.exp_config['model']['freeze_layers'],
            'weight_decay': self.exp_config['training']['weight_decay'],
            'train_acc': metrics['train_acc'],
            'test_acc': metrics['test_acc'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'train_time_sec': train_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'notes': self.exp_config['experiment'].get('notes', '')
        }

        # 追加到文件
        df = pd.DataFrame([record])
        if tracker_file.exists():
            df.to_csv(tracker_file, mode='a', header=False, index=False)
        else:
            df.to_csv(tracker_file, index=False)

        self.logger.info(f"实验记录已更新")

    def run(self):
        """运行实验"""
        start_time = time.time()

        try:
            # 1. 加载数据
            X_train, X_test, y_train, y_test = self.load_data()

            # 2. 创建数据加载器
            tokenizer, train_loader, test_loader = self.create_dataloaders(
                X_train, X_test, y_train, y_test
            )

            # 3. 创建模型
            model = self.create_model()

            # 4. 创建优化器和调度器
            optimizer, scheduler = self.create_optimizer_scheduler(model, train_loader)

            # 5. 训练
            self.logger.info("开始训练...")
            self.logger.info(f"学习率: {self.exp_config['training']['learning_rate']}")
            self.logger.info(f"批次大小: {self.exp_config['training']['batch_size']}")
            self.logger.info(f"训练轮数: {self.exp_config['training']['num_epochs']}")

            num_epochs = self.exp_config['training']['num_epochs']

            for epoch in range(num_epochs):
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, scheduler, epoch
                )

                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                )

            # 6. 评估
            self.logger.info("开始评估...")
            y_true, y_pred = self.evaluate(model, test_loader)

            # 计算指标
            test_acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')

            metrics = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted
            }

            self.logger.info(f"测试准确率: {test_acc:.4f}")
            self.logger.info(f"F1 (macro): {f1_macro:.4f}")
            self.logger.info(f"F1 (weighted): {f1_weighted:.4f}")

            # 7. 保存结果
            train_time = time.time() - start_time
            self.save_results(y_true, y_pred, metrics, train_time)

            # 8. 保存模型（可选）
            if self.exp_config['experiment']['save_model']:
                model_dir = self.exp_dir / 'model'
                model_dir.mkdir(exist_ok=True)

                utils.save_model_with_metadata(
                    model, tokenizer,
                    metadata=metrics,
                    save_dir=str(model_dir)
                )
                self.logger.info(f"模型已保存到: {model_dir}")

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"实验完成！耗时: {train_time/60:.2f} 分钟")
            self.logger.info(f"结果保存在: {self.exp_dir}")
            self.logger.info(f"{'='*60}\n")

            return metrics

        except Exception as e:
            self.logger.error(f"实验失败: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description='运行BERT微调实验')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='实验配置文件路径'
    )

    args = parser.parse_args()

    # 运行实验
    runner = ExperimentRunner(args.config)
    runner.run()


if __name__ == '__main__':
    main()
