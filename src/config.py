# src/config.py

import os

# ------------------------
# 数据路径
# ------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "data_preprocessed_python")

# 验证路径是否存在
if not os.path.exists(DATA_DIR):
    print(f"Warning: DATA_DIR {DATA_DIR} does not exist")

# ------------------------
# GNN 配置
# ------------------------
GRAPH_TYPE = "plv"       # 'pearson' / 'plv'
THRESH = 0.5             # 邻接矩阵阈值
NODE_FEATURE_DIM = 5
OUTPUT_DIM = 4           # DEAP 四维情绪

# ------------------------
# 训练超参数
# ------------------------
MODEL_TYPE = "gcn"       # 'gcn' / 'gat'
HIDDEN_DIM = 32
DROPOUT = 0.5
HEADS = 4                # GAT 多头注意力
LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 50

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
