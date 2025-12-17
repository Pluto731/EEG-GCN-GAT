# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.graph.build_graph import EEGGraphDataset
from src.models.gcn import GCN
from src.models.gat import GAT
import numpy as np

# ------------------------
# 1️⃣ 模拟加载数据 (替换为实际 DEAP segment)
# ------------------------
# 假设已经加载 segments: (N, 32, T), labels: (N, 4)
# 这里用随机数据演示
np.random.seed(0)
N_samples = 100
segments = np.random.randn(N_samples, 32, 256)
labels = np.random.rand(N_samples, 4)

# 构建 PyG Dataset
dataset = EEGGraphDataset(segments, labels, graph_type=GRAPH_TYPE, thresh=THRESH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------
# 2️⃣ 初始化模型
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_TYPE.lower() == "gcn":
    model = GCN(in_channels=NODE_FEATURE_DIM, hidden_channels=HIDDEN_DIM, out_channels=OUTPUT_DIM, dropout=DROPOUT)
elif MODEL_TYPE.lower() == "gat":
    model = GAT(in_channels=NODE_FEATURE_DIM, hidden_channels=HIDDEN_DIM, out_channels=OUTPUT_DIM, heads=HEADS, dropout=DROPOUT)
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()  # DEAP 四维情绪回归任务

# ------------------------
# 3️⃣ 训练循环
# ------------------------
best_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # out = model(batch)
        # loss = criterion(out, batch.y)
        out = model(batch)                 # [batch_size, 4]
        y = batch.y.view(out.size(0), -1)  # [batch_size, 4]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {epoch_loss:.4f}")

    # 保存最优模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        save_path = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_best.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  Saved best model to {save_path}")
