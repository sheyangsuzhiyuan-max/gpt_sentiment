#!/bin/bash
# 批量运行所有实验配置

echo "======================================"
echo "  批量实验运行脚本"
echo "======================================"
echo ""

# 确保在项目根目录
cd "$(dirname "$0")/.."

# 实验配置列表
configs=(
    "experiments/configs/baseline.yaml"
    "experiments/configs/lower_lr.yaml"
    "experiments/configs/higher_lr.yaml"
    "experiments/configs/freeze_layers.yaml"
    "experiments/configs/heavy_freeze.yaml"
)

# 运行每个实验
for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "======================================"
        echo "运行实验: $config"
        echo "======================================"
        python experiments/run_experiment.py --config "$config"

        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo "✅ 实验完成: $config"
        else
            echo "❌ 实验失败: $config"
        fi
        echo ""
    else
        echo "⚠️  配置文件不存在: $config"
    fi
done

echo "======================================"
echo "  所有实验完成！"
echo "======================================"
echo ""
echo "运行以下命令查看对比结果:"
echo "  python experiments/compare_experiments.py"
echo ""
