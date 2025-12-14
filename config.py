"""
项目配置文件
统一管理所有超参数和路径配置
"""
import os
import torch

# ============= 随机种子 =============
RANDOM_SEED = 42

# ============= 路径配置 =============
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model_save')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 数据文件
RAW_DATA_FILE = 'raw_data.csv'
PROCESSED_DATA_FILE = 'processed_data.csv'

# ============= 数据处理配置 =============
# 数据列名
TEXT_COLUMN = 'tweets'
LABEL_COLUMN = 'labels'
CLEANED_TEXT_COLUMN = 'cleaned_text'

# 标签映射
LABEL_MAP = {'bad': 0, 'neutral': 1, 'good': 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

# 数据划分
TEST_SIZE = 0.2

# 文本清洗
MAX_TEXT_LENGTH = 1000  # 最大输入文本长度限制

# ============= 模型配置 =============
# BERT 配置
BERT_MODEL_NAME = 'distilbert-base-uncased'
MAX_SEQ_LENGTH = 128
BERT_BATCH_SIZE = 32
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 1

# RNN 配置
RNN_VOCAB_MIN_FREQ = 2  # 词表最小词频
RNN_MAX_LEN = 50
RNN_EMBED_DIM = 100
RNN_HIDDEN_DIM = 128
RNN_BATCH_SIZE = 64
RNN_LEARNING_RATE = 0.001
RNN_EPOCHS = 5

# Baseline 配置
TFIDF_MAX_FEATURES = 5000
LOGISTIC_MAX_ITER = 1000

# ============= 设备配置 =============
def get_device():
    """自动选择最优设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# ============= 训练配置 =============
# 日志打印频率（每N个batch打印一次）
LOG_INTERVAL = 100

# MPS 缓存清理频率
CACHE_CLEAR_INTERVAL = 1000

# ============= Streamlit App 配置 =============
APP_TITLE = "Voice of User | AI Analysis"
APP_ICON = "🤖"
APP_LAYOUT = "wide"

# 模型信息
MODEL_INFO = {
    'course': 'CA6001',
    'model_name': 'DistilBERT (Fine-tuned)',
    'accuracy': '93%',
}

# 停用词表（用于关键词提取）
STOPWORDS = {
    'the', 'is', 'a', 'to', 'and', 'of', 'it', 'but', 'for', 'in',
    'my', 'i', 'this', 'that', 'with', 'on', 'be', 'are', 'was',
    'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can'
}

# ============= 可视化配置 =============
SENTIMENT_COLORS = {
    'Bad': '#ff4b4b',
    'Neutral': '#ffa421',
    'Good': '#21c354'
}

# ============= 创建必要目录 =============
def create_dirs():
    """创建项目必要的目录"""
    for dir_path in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

if __name__ == '__main__':
    create_dirs()
    print(f"✅ 目录创建完成")
    print(f"📂 数据目录: {DATA_DIR}")
    print(f"📂 模型目录: {MODEL_DIR}")
    print(f"📂 日志目录: {LOGS_DIR}")
    print(f"🔥 运行设备: {DEVICE}")
