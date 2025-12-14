"""
工具函数模块
包含数据处理、模型训练、评估等通用函数
"""
import os
import re
import random
import logging
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import config

# ============= 日志配置 =============
def setup_logger(name='sentiment_analysis', log_file=None):
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，如果为None则只输出到控制台

    Returns:
        logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除已有的handlers
    logger.handlers.clear()

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        config.create_dirs()
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============= 随机种子设置 =============
def set_seed(seed=config.RANDOM_SEED):
    """
    设置所有随机种子，确保结果可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============= 路径处理 =============
def find_data_file(filename):
    """
    统一的数据文件查找函数

    Args:
        filename: 文件名

    Returns:
        文件的完整路径

    Raises:
        FileNotFoundError: 如果文件不存在
    """
    # 可能的路径列表
    possible_paths = [
        os.path.join(config.DATA_DIR, filename),  # 根目录/data/
        os.path.join('..', 'data', filename),      # notebooks/../data/
        os.path.join('data', filename),            # 当前目录/data/
        filename                                   # 当前目录
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"找不到文件 '{filename}'。请确保文件在以下位置之一:\n" +
        "\n".join(f"  - {p}" for p in possible_paths)
    )


# ============= 数据清洗 =============
def clean_text(text):
    """
    清洗文本数据

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not isinstance(text, str):
        return ""

    # 转小写
    text = text.lower()

    # 去除 URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 去除 @用户名 和 #标签
    text = re.sub(r'\@\w+|\#\w+', '', text)

    # 去除标点符号（保留空格）
    text = re.sub(r'[^\w\s]', '', text)

    # 去除多余空格
    text = " ".join(text.split())

    return text


def remove_stopwords(text, stopwords=config.STOPWORDS):
    """
    移除停用词

    Args:
        text: 文本
        stopwords: 停用词集合

    Returns:
        移除停用词后的文本
    """
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords]
    return " ".join(filtered_words)


# ============= 数据验证 =============
def validate_dataframe(df, required_columns):
    """
    验证DataFrame是否包含必要的列

    Args:
        df: pandas DataFrame
        required_columns: 必需的列名列表

    Raises:
        ValueError: 如果缺少必需的列
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"数据缺少必需的列: {missing_cols}\n"
            f"当前列: {df.columns.tolist()}"
        )


# ============= PyTorch Dataset =============
class SentimentDataset(Dataset):
    """
    情感分析数据集类（用于BERT）

    Args:
        texts: 文本列表或Series
        labels: 标签列表或Series
        tokenizer: BERT tokenizer
        max_len: 最大序列长度
    """
    def __init__(self, texts, labels, tokenizer, max_len=config.MAX_SEQ_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RNNTextDataset(Dataset):
    """
    RNN文本数据集类

    Args:
        texts: 文本Series
        labels: 标签Series
        vocab: 词表字典
        max_len: 最大序列长度
    """
    def __init__(self, texts, labels, vocab=None, max_len=config.RNN_MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        # 如果没传词表，就根据当前数据建立
        if vocab is None:
            all_words = " ".join(texts).split()
            word_counts = Counter(all_words)
            # 只保留出现超过指定次数的词
            vocab = {
                word: i+2
                for i, (word, count) in enumerate(word_counts.items())
                if count > config.RNN_VOCAB_MIN_FREQ
            }
            vocab['<PAD>'] = 0
            vocab['<UNK>'] = 1

        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        # 文本转数字序列
        tokens = [self.vocab.get(word, 1) for word in text.split()]

        # 截断或填充
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ============= 模型保存与加载 =============
def save_model_with_metadata(model, tokenizer, metadata, save_dir=config.MODEL_DIR):
    """
    保存模型并附带元数据

    Args:
        model: 模型实例
        tokenizer: tokenizer实例
        metadata: 元数据字典（如准确率、训练时间等）
        save_dir: 保存目录
    """
    config.create_dirs()

    # 保存模型
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 保存元数据
    metadata_file = os.path.join(save_dir, 'metadata.txt')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    return save_dir


# ============= 关键词提取 =============
def extract_keywords(text, top_n=3, stopwords=config.STOPWORDS):
    """
    提取文本中的关键词

    Args:
        text: 输入文本
        top_n: 返回前N个关键词
        stopwords: 停用词集合

    Returns:
        关键词列表及其频次
    """
    words = re.findall(r'\w+', text.lower())
    # 筛选：不在停用词表且长度大于3的词
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    return Counter(keywords).most_common(top_n)


# ============= 训练辅助函数 =============
def clear_mps_cache():
    """清理MPS缓存"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def clear_cuda_cache():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device_info():
    """获取设备信息"""
    device = config.DEVICE
    if device.type == 'mps':
        return "Apple Silicon (MPS)"
    elif device.type == 'cuda':
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU"


# ============= 输入验证 =============
def validate_input_text(text, max_length=config.MAX_TEXT_LENGTH):
    """
    验证并清理用户输入文本

    Args:
        text: 输入文本
        max_length: 最大长度

    Returns:
        (是否有效, 清理后的文本, 警告信息)
    """
    if not text or not text.strip():
        return False, "", "输入不能为空"

    text = text.strip()

    if len(text) > max_length:
        warning = f"输入文本过长（{len(text)}字符），已截断至{max_length}字符"
        text = text[:max_length]
        return True, text, warning

    return True, text, None


if __name__ == '__main__':
    # 测试功能
    print("=" * 50)
    print("工具模块测试")
    print("=" * 50)

    # 测试随机种子
    set_seed()
    print(f"✅ 随机种子已设置: {config.RANDOM_SEED}")

    # 测试设备信息
    print(f"✅ 设备信息: {get_device_info()}")

    # 测试文本清洗
    test_text = "Check out https://example.com! @user #hashtag THIS is A Test!!!"
    cleaned = clean_text(test_text)
    print(f"✅ 文本清洗测试:")
    print(f"   原文: {test_text}")
    print(f"   清洗后: {cleaned}")

    # 测试关键词提取
    keywords = extract_keywords("machine learning is amazing and machine learning is powerful")
    print(f"✅ 关键词提取: {keywords}")

    print("\n所有测试通过!")
