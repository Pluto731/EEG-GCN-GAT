# src/config.py

import os

# ------------------------
# 数据路径
# ------------------------
# 获取当前文件所在目录 (src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 由于 config.py 在 src 目录下，需要向上一级到达项目根目录
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 构建数据目录路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "data_preprocessed_python")

# 打印调试信息
print(f"CURRENT_DIR: {CURRENT_DIR}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"DATA_DIR exists: {os.path.exists(DATA_DIR)}")

# 验证路径是否存在
if not os.path.exists(DATA_DIR):
    print(f"Warning: DATA_DIR {DATA_DIR} does not exist")

# ------------------------
# GNN 配置
# ------------------------
GRAPH_TYPE = "pearson"       # 'pearson' / 'plv'
THRESH = 0.5             # 邻接矩阵阈值
NODE_FEATURE_DIM = 5
OUTPUT_DIM = 4           # DEAP 四维情绪

# ------------------------
# 训练超参数
# ------------------------
MODEL_TYPE = "gat"       # 'gcn' / 'gat'
HIDDEN_DIM = 32
DROPOUT = 0.5
HEADS = 4                # GAT 多头注意力
LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 10            # 早停耐心值   

# ------------------------
# 断点续训参数
# ------------------------
RESUME_TRAINING = False  # 是否从检查点继续训练
CHECKPOINT_INTERVAL = 5  # 每隔多少个epoch保存一次检查点

# ------------------------
# 保存路径
# ------------------------
SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------
# 配置验证
# ------------------------
assert GRAPH_TYPE in ['pearson', 'plv'], "GRAPH_TYPE must be 'pearson' or 'plv'"
assert MODEL_TYPE in ['gcn', 'gat'], "MODEL_TYPE must be 'gcn' or 'gat'"
assert 0 <= DROPOUT <= 1, "DROPOUT must be between 0 and 1"