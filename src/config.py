# src/config.py

import os

# ------------------------
# 1. 路径配置 (Path Configuration)
# ------------------------
# 获取当前文件所在目录 (.../src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 (.../LastWork/)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 原始数据目录 (DEAP 官方提供的 data_preprocessed_python 文件夹)
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "data_preprocessed_python")

# 处理后的图数据保存目录 (存放 .pt 文件)
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# 检查点保存目录
SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# 打印调试信息
print(f"[Config] Project Root: {PROJECT_ROOT}")
print(f"[Config] Raw Data Dir: {DATA_RAW_DIR} (Exists: {os.path.exists(DATA_RAW_DIR)})")
print(f"[Config] Processed Dir: {DATA_PROCESSED_DIR}")

# ------------------------
# 2. 信号预处理参数 (Preprocessing)
# ------------------------
FS = 128                  # DEAP 采样率
WINDOW_SIZE = 128         # 时间窗口长度 (128点 = 1秒)
STEP_SIZE = 128           # 滑动步长 (128点 = 无重叠; 64点 = 50%重叠)
BASELINE_LEN = 3 * 128    # 基线长度 (前3秒通常去除)

# ------------------------
# 3. GNN 构图配置 (Graph Construction)
# ------------------------
GRAPH_TYPE = "pearson"    # 'pearson' 或 'plv'
THRESHOLD = 0.5           # 邻接矩阵保留连接的阈值
NODE_FEATURE_DIM = 5      # 节点特征维度 (DE: Delta, Theta, Alpha, Beta, Gamma)

# ------------------------
# 4. 标签配置 (Label)
# ------------------------
OUTPUT_DIM = 4            # 输出维度 (Valence, Arousal, Dominance, Liking)
LABEL_TYPE = "binary"     # 'binary' (二分类) 或 'regression' (回归)
LABEL_THRESHOLD = 5.0     # 二分类阈值 (DEAP 评分范围 1-9，通常以 5 分界)

# ------------------------
# 5. 模型与训练超参数 (Model & Training)
# ------------------------
MODEL_TYPE = "gcn"        # 'gcn' 或 'gat'
HIDDEN_DIM = 64           # 隐藏层维度
DROPOUT = 0.5             # Dropout
HEADS = 4                 # GAT 多头注意力数量
LR = 0.001                # 学习率
BATCH_SIZE = 64           # 批次大小
EPOCHS = 100              # 训练轮数
PATIENCE = 20             # 早停耐心值

# ------------------------
# 6. 断点续训 (Resume)
# ------------------------
RESUME_TRAINING = False   # 是否从检查点继续训练
CHECKPOINT_INTERVAL = 10  # 每隔多少个epoch保存一次检查点

# ------------------------
# 配置验证 (Assertions)
# ------------------------
if not os.path.exists(DATA_RAW_DIR):
    print(f"Warning: Raw data directory {DATA_RAW_DIR} does not exist. Please check path.")

assert GRAPH_TYPE in ['pearson', 'plv'], "GRAPH_TYPE must be 'pearson' or 'plv'"
assert MODEL_TYPE in ['gcn', 'gat'], "MODEL_TYPE must be 'gcn' or 'gat'"
assert 0 <= DROPOUT <= 1, "DROPOUT must be between 0 and 1"