"""
实验对比脚本
可视化对比不同实验的结果
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_experiments():
    """加载所有实验记录"""
    tracker_file = Path('experiments/experiment_tracker.csv')

    if not tracker_file.exists():
        print("❌ 未找到实验记录文件")
        return None

    df = pd.read_csv(tracker_file)
    print(f"✅ 加载了 {len(df)} 个实验记录\n")

    return df


def print_summary_table(df):
    """打印汇总表"""
    print("=" * 100)
    print("实验汇总表")
    print("=" * 100)

    # 选择关键列
    summary_cols = [
        'exp_id', 'name', 'learning_rate', 'freeze_layers',
        'train_acc', 'test_acc', 'f1_macro'
    ]

    summary = df[summary_cols].copy()

    # 格式化
    summary['train_acc'] = summary['train_acc'].apply(lambda x: f"{x:.4f}")
    summary['test_acc'] = summary['test_acc'].apply(lambda x: f"{x:.4f}")
    summary['f1_macro'] = summary['f1_macro'].apply(lambda x: f"{x:.4f}")
    summary['learning_rate'] = summary['learning_rate'].apply(lambda x: f"{x:.2e}")

    print(summary.to_string(index=False))
    print("=" * 100)
    print()


def find_best_experiment(df):
    """找出最佳实验"""
    print("=" * 100)
    print("最佳实验")
    print("=" * 100)

    # 按测试准确率排序
    best_acc = df.loc[df['test_acc'].idxmax()]

    print(f"🏆 最高测试准确率:")
    print(f"   实验ID: {best_acc['exp_id']}")
    print(f"   名称: {best_acc['name']}")
    print(f"   准确率: {best_acc['test_acc']:.4f}")
    print(f"   F1-score: {best_acc['f1_macro']:.4f}")
    print(f"   学习率: {best_acc['learning_rate']:.2e}")
    print(f"   冻结层数: {int(best_acc['freeze_layers'])}")
    print()

    # 按F1排序
    best_f1 = df.loc[df['f1_macro'].idxmax()]

    if best_f1['exp_id'] != best_acc['exp_id']:
        print(f"🥈 最高F1-score:")
        print(f"   实验ID: {best_f1['exp_id']}")
        print(f"   名称: {best_f1['name']}")
        print(f"   F1-score: {best_f1['f1_macro']:.4f}")
        print()

    print("=" * 100)
    print()


def plot_comparisons(df):
    """绘制对比图"""
    print("生成对比图...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 准确率对比
    ax1 = axes[0, 0]
    x = range(len(df))
    width = 0.35

    ax1.bar([i - width/2 for i in x], df['train_acc'], width,
            label='Train Acc', alpha=0.8, color='#3498db')
    ax1.bar([i + width/2 for i in x], df['test_acc'], width,
            label='Test Acc', alpha=0.8, color='#e74c3c')

    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Testing Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['exp_id'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. F1-score 对比
    ax2 = axes[0, 1]
    ax2.plot(x, df['f1_macro'], marker='o', label='F1 (macro)',
             linewidth=2, markersize=8, color='#2ecc71')
    ax2.plot(x, df['f1_weighted'], marker='s', label='F1 (weighted)',
             linewidth=2, markersize=8, color='#9b59b6')

    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['exp_id'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. 学习率 vs 准确率
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['learning_rate'], df['test_acc'],
                          s=100, c=df['freeze_layers'],
                          cmap='viridis', alpha=0.7, edgecolors='black')

    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Learning Rate vs Test Accuracy')
    ax3.set_xscale('log')
    ax3.grid(alpha=0.3)

    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Freeze Layers')

    # 标注实验ID
    for idx, row in df.iterrows():
        ax3.annotate(row['exp_id'].replace('exp_', ''),
                     (row['learning_rate'], row['test_acc']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.7)

    # 4. 训练时间对比
    ax4 = axes[1, 1]
    train_time_min = df['train_time_sec'] / 60

    bars = ax4.bar(x, train_time_min, alpha=0.7, color='#f39c12')
    ax4.set_xlabel('Experiment')
    ax4.set_ylabel('Training Time (minutes)')
    ax4.set_title('Training Time Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['exp_id'], rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)

    # 在柱子上显示数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{train_time_min.iloc[i]:.1f}m',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存图表
    output_file = Path('experiments/experiment_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图已保存: {output_file}")

    plt.show()


def analyze_hyperparameters(df):
    """分析超参数影响"""
    print("=" * 100)
    print("超参数影响分析")
    print("=" * 100)

    # 学习率影响
    print("\n📊 学习率影响:")
    lr_groups = df.groupby('learning_rate')['test_acc'].agg(['mean', 'max', 'count'])
    print(lr_groups)

    # 冻结层数影响
    print("\n📊 冻结层数影响:")
    freeze_groups = df.groupby('freeze_layers')['test_acc'].agg(['mean', 'max', 'count'])
    print(freeze_groups)

    # 权重衰减影响
    if 'weight_decay' in df.columns and df['weight_decay'].nunique() > 1:
        print("\n📊 权重衰减影响:")
        wd_groups = df.groupby('weight_decay')['test_acc'].agg(['mean', 'max', 'count'])
        print(wd_groups)

    print("=" * 100)
    print()


def generate_report(df):
    """生成markdown报告"""
    report_file = Path('experiments/EXPERIMENT_REPORT.md')

    with open(report_file, 'w') as f:
        f.write("# 实验报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 实验总结\n\n")
        f.write(f"- 总实验数: {len(df)}\n")
        f.write(f"- 最高准确率: {df['test_acc'].max():.4f}\n")
        f.write(f"- 最高F1: {df['f1_macro'].max():.4f}\n")
        f.write(f"- 平均准确率: {df['test_acc'].mean():.4f}\n\n")

        f.write("## 详细结果\n\n")
        f.write("| 实验ID | 名称 | 学习率 | 冻结层 | 测试准确率 | F1-score |\n")
        f.write("|--------|------|--------|---------|------------|----------|\n")

        for _, row in df.iterrows():
            f.write(f"| {row['exp_id']} | {row['name']} | "
                   f"{row['learning_rate']:.2e} | {int(row['freeze_layers'])} | "
                   f"{row['test_acc']:.4f} | {row['f1_macro']:.4f} |\n")

        f.write("\n## 最佳配置\n\n")
        best = df.loc[df['test_acc'].idxmax()]
        f.write(f"- **实验ID**: {best['exp_id']}\n")
        f.write(f"- **名称**: {best['name']}\n")
        f.write(f"- **学习率**: {best['learning_rate']:.2e}\n")
        f.write(f"- **批次大小**: {int(best['batch_size'])}\n")
        f.write(f"- **冻结层数**: {int(best['freeze_layers'])}\n")
        f.write(f"- **权重衰减**: {best['weight_decay']}\n")
        f.write(f"- **测试准确率**: {best['test_acc']:.4f}\n")
        f.write(f"- **F1-score**: {best['f1_macro']:.4f}\n\n")

        f.write("## 结论\n\n")
        f.write("（根据实验结果填写你的发现和结论）\n\n")

    print(f"✅ 实验报告已生成: {report_file}")


def main():
    """主函数"""
    print("\n" + "=" * 100)
    print(" " * 35 + "实验结果对比分析")
    print("=" * 100 + "\n")

    # 加载实验记录
    df = load_experiments()

    if df is None or len(df) == 0:
        print("没有实验记录")
        return

    # 打印汇总表
    print_summary_table(df)

    # 找出最佳实验
    find_best_experiment(df)

    # 分析超参数影响
    analyze_hyperparameters(df)

    # 绘制对比图
    plot_comparisons(df)

    # 生成报告
    generate_report(df)

    print("\n" + "=" * 100)
    print(" " * 40 + "分析完成！")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
