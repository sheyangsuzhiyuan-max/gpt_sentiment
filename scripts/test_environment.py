"""
环境测试脚本
验证所有依赖是否正确安装
"""
import sys

def test_imports():
    """测试所有必需的包是否可以导入"""
    print("=" * 60)
    print("测试环境配置")
    print("=" * 60)

    packages = {
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'streamlit': 'Streamlit',
        'altair': 'Altair',
        'nltk': 'NLTK',
    }

    failed = []

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name:20s} - 已安装")
        except ImportError:
            print(f"❌ {name:20s} - 未安装")
            failed.append(name)

    print("\n" + "=" * 60)

    if failed:
        print(f"❌ 以下包未安装: {', '.join(failed)}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有必需包已安装")
        return True


def test_versions():
    """显示关键包的版本"""
    print("\n" + "=" * 60)
    print("包版本信息")
    print("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        import torch
        import transformers
        import sklearn
        import streamlit as st

        print(f"Python:        {sys.version.split()[0]}")
        print(f"Pandas:        {pd.__version__}")
        print(f"NumPy:         {np.__version__}")
        print(f"PyTorch:       {torch.__version__}")
        print(f"Transformers:  {transformers.__version__}")
        print(f"scikit-learn:  {sklearn.__version__}")
        print(f"Streamlit:     {st.__version__}")

    except Exception as e:
        print(f"❌ 获取版本信息失败: {e}")
        return False

    return True


def test_device():
    """测试可用的计算设备"""
    print("\n" + "=" * 60)
    print("设备信息")
    print("=" * 60)

    try:
        import torch

        # CPU
        print(f"CPU:           可用")

        # CUDA
        if torch.cuda.is_available():
            print(f"CUDA:          可用 ({torch.cuda.get_device_name(0)})")
        else:
            print(f"CUDA:          不可用")

        # MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print(f"MPS:           可用 (Apple Silicon)")
            print(f"推荐设备:      MPS (GPU加速)")
        else:
            print(f"MPS:           不可用")
            if torch.cuda.is_available():
                print(f"推荐设备:      CUDA")
            else:
                print(f"推荐设备:      CPU")

    except Exception as e:
        print(f"❌ 检测设备失败: {e}")
        return False

    return True


def test_config_utils():
    """测试项目配置和工具模块"""
    print("\n" + "=" * 60)
    print("项目配置测试")
    print("=" * 60)

    try:
        import config
        import utils

        print(f"✅ config.py   - 导入成功")
        print(f"✅ utils.py    - 导入成功")

        # 测试关键配置
        print(f"\n关键配置:")
        print(f"  随机种子:    {config.RANDOM_SEED}")
        print(f"  设备:        {config.DEVICE}")
        print(f"  模型目录:    {config.MODEL_DIR}")
        print(f"  数据目录:    {config.DATA_DIR}")

        # 测试工具函数
        utils.set_seed()
        print(f"\n✅ 工具函数测试通过")

    except ImportError as e:
        print(f"❌ 导入项目模块失败: {e}")
        print("\n提示: 确保在项目根目录运行此脚本")
        return False
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

    return True


def test_data_files():
    """检查数据文件"""
    print("\n" + "=" * 60)
    print("数据文件检查")
    print("=" * 60)

    try:
        import config
        import utils
        import os

        # 检查原始数据
        try:
            raw_path = utils.find_data_file(config.RAW_DATA_FILE)
            print(f"✅ 原始数据:   {raw_path}")
        except FileNotFoundError:
            print(f"⚠️  原始数据:   未找到 ({config.RAW_DATA_FILE})")

        # 检查处理后数据
        try:
            processed_path = utils.find_data_file(config.PROCESSED_DATA_FILE)
            print(f"✅ 处理数据:   {processed_path}")
        except FileNotFoundError:
            print(f"⚠️  处理数据:   未找到 ({config.PROCESSED_DATA_FILE})")
            print(f"   提示: 运行 01_EDA_Preprocess.ipynb 生成")

        # 检查模型文件
        if os.path.exists(config.MODEL_DIR):
            model_files = os.listdir(config.MODEL_DIR)
            if model_files:
                print(f"✅ 模型文件:   已找到 ({len(model_files)} 个文件)")
            else:
                print(f"⚠️  模型文件:   目录为空")
        else:
            print(f"⚠️  模型文件:   未找到模型目录")
            print(f"   提示: 运行 04_BERT_Finetune.ipynb 训练模型")

    except Exception as e:
        print(f"❌ 数据文件检查失败: {e}")
        return False

    return True


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "环境测试脚本" + " " * 31 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    tests = [
        ("依赖包导入", test_imports),
        ("包版本检查", test_versions),
        ("设备检测", test_device),
        ("项目配置", test_config_utils),
        ("数据文件", test_data_files),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 测试失败: {e}")
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15s}: {status}")

    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 项测试通过")

    if passed == total:
        print("\n🎉 环境配置完美！可以开始使用项目了。")
        print("\n下一步:")
        print("  1. 运行 notebooks/01_EDA_Preprocess.ipynb")
        print("  2. 运行 notebooks/04_BERT_Finetune.ipynb")
        print("  3. 运行 streamlit run app.py")
    else:
        print("\n⚠️  部分测试未通过，请检查上述错误信息。")

    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
