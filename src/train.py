# src/train.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm  # 训练进度条
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.graph.build_graph import EEGGraphDataset
from src.models.gcn import GCN
from src.models.gat import GAT
from src.data.load_deap import DEAPLoader

def save_checkpoint(model, optimizer, epoch, train_losses, train_mae, val_losses, val_mae, best_val_loss, patience_counter, save_path):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_mae': train_mae,
        'val_losses': val_losses,
        'val_mae': val_mae,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, save_path):
    """加载训练检查点"""
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (
        checkpoint['epoch'], 
        checkpoint['train_losses'], 
        checkpoint['train_mae'], 
        checkpoint['val_losses'], 
        checkpoint['val_mae'], 
        checkpoint['best_val_loss'], 
        checkpoint['patience_counter']
    )

def get_latest_checkpoint():
    """获取最新的检查点文件"""
    checkpoint_dir = os.path.join(SAVE_DIR, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{MODEL_TYPE}_checkpoint_epoch_")]
    if not checkpoints:
        return None
    
    # 按epoch数排序，获取最大的
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_checkpoint = f"{MODEL_TYPE}_checkpoint_epoch_{latest_epoch}.pt"
    return os.path.join(checkpoint_dir, latest_checkpoint)

# ----------------------------
# 1️⃣ 加载数据
# ----------------------------
print(f"DATA_DIR exists: {os.path.exists(DATA_DIR)}")
loader = DEAPLoader(DATA_DIR)
X, y, subjects = loader.load_all_subjects()

# 滑动窗口分段
segment_length = 256
stride = 128
segments_list, labels_list, subject_ids_list = [], [], []

for trial_idx in range(X.shape[0]):
    trial = X[trial_idx]
    label = y[trial_idx]
    subject_id = subjects[trial_idx]
    for start in range(0, trial.shape[1] - segment_length + 1, stride):
        segment = trial[:, start:start+segment_length]
        segments_list.append(segment)
        labels_list.append(label)
        subject_ids_list.append(subject_id)

segments = np.array(segments_list, dtype=np.float32)  # 默认 float64 太大
labels = np.stack(labels_list)
subject_ids = np.array(subject_ids_list)
print(f"Total segments: {segments.shape[0]}")

# 数据标准化
mean = segments.mean(axis=0)
std = segments.std(axis=0)
segments = (segments - mean) / (std + 1e-8)

# 分割数据集 (70% 训练, 15% 验证, 15% 测试)
indices = np.arange(len(segments))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# 构建 PyG Dataset
train_dataset = EEGGraphDataset(segments[train_idx], labels[train_idx], graph_type=GRAPH_TYPE, thresh=THRESH)
val_dataset = EEGGraphDataset(segments[val_idx], labels[val_idx], graph_type=GRAPH_TYPE, thresh=THRESH)
test_dataset = EEGGraphDataset(segments[test_idx], labels[test_idx], graph_type=GRAPH_TYPE, thresh=THRESH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}, Test set: {len(test_dataset)}")

# ----------------------------
# 2️⃣ 初始化模型
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL_TYPE.lower() == "gcn":
    model = GCN(NODE_FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
elif MODEL_TYPE.lower() == "gat":
    model = GAT(NODE_FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM, heads=HEADS, dropout=DROPOUT)
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ----------------------------
# 3️⃣ 断点续训处理
# ----------------------------
start_epoch = 0
train_losses, train_mae, val_losses, val_mae = [], [], [], []
best_val_loss = float("inf")
patience_counter = 0

# 如果启用了断点续训并且存在检查点，则加载
if RESUME_TRAINING:
    latest_checkpoint_path = get_latest_checkpoint()
    if latest_checkpoint_path:
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        start_epoch, train_losses, train_mae, val_losses, val_mae, best_val_loss, patience_counter = \
            load_checkpoint(model, optimizer, latest_checkpoint_path)
        print(f"Resuming from epoch {start_epoch + 1}, best validation loss: {best_val_loss:.4f}")
    else:
        print("No checkpoint found, starting from scratch.")
else:
    print("Starting training from scratch.")

# 创建检查点目录
checkpoint_dir = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# ----------------------------
# 4️⃣ 训练循环 + 可视化记录
# ----------------------------
for epoch in range(start_epoch + 1, EPOCHS+1):
    # 训练阶段
    model.train()
    epoch_loss, epoch_mae = 0, 0

    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        y_true = batch.y.view(out.size(0), -1)
        loss = criterion(out, y_true)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch.num_graphs
        epoch_mae += torch.mean(torch.abs(out - y_true)).item() * batch.num_graphs

    epoch_loss /= len(train_loader.dataset)
    epoch_mae /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_mae.append(epoch_mae)

    # 验证阶段
    model.eval()
    val_epoch_loss, val_epoch_mae = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            y_true = batch.y.view(out.size(0), -1)
            loss = criterion(out, y_true)
            
            val_epoch_loss += loss.item() * batch.num_graphs
            val_epoch_mae += torch.mean(torch.abs(out - y_true)).item() * batch.num_graphs

    val_epoch_loss /= len(val_loader.dataset)
    val_epoch_mae /= len(val_loader.dataset)
    val_losses.append(val_epoch_loss)
    val_mae.append(val_epoch_mae)

    print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {epoch_loss:.4f} | Train MAE: {epoch_mae:.4f} | "
          f"Val Loss: {val_epoch_loss:.4f} | Val MAE: {val_epoch_mae:.4f}")

    # 保存最优模型
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        patience_counter = 0
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_best.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  Saved best model to {save_path}")
    else:
        patience_counter += 1

    # 保存检查点（根据CHECKPOINT_INTERVAL参数决定是否保存）
    if epoch % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"{MODEL_TYPE}_checkpoint_epoch_{epoch}.pt")
        save_checkpoint(
            model, optimizer, epoch, train_losses, train_mae, 
            val_losses, val_mae, best_val_loss, patience_counter, checkpoint_path
        )
        print(f"  Saved checkpoint to {checkpoint_path}")

    # 早停机制
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# 保存最终的检查点
final_checkpoint_path = os.path.join(checkpoint_dir, f"{MODEL_TYPE}_final_checkpoint.pt")
save_checkpoint(
    model, optimizer, epoch, train_losses, train_mae, 
    val_losses, val_mae, best_val_loss, patience_counter, final_checkpoint_path
)
print(f"Saved final checkpoint to {final_checkpoint_path}")

# ----------------------------
# 5️⃣ Loss & MAE 曲线可视化
# ----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{MODEL_TYPE.upper()} Training & Validation Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_mae)+1), train_mae, marker='x', label='Train MAE')
plt.plot(range(1, len(val_mae)+1), val_mae, marker='x', label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title(f'{MODEL_TYPE.upper()} Training & Validation MAE')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# 6️⃣ 测试集评估
# ----------------------------
model.eval()
test_loss, test_mae = 0, 0
all_preds, all_targets = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        y_true = batch.y.view(out.size(0), -1)
        loss = criterion(out, y_true)
        mae = torch.mean(torch.abs(out - y_true))
        
        test_loss += loss.item() * batch.num_graphs
        test_mae += mae.item() * batch.num_graphs
        all_preds.append(out.cpu().numpy())
        all_targets.append(y_true.cpu().numpy())

test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)
all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# ----------------------------
# 7️⃣ 预测 vs 真实标签对比
# ----------------------------
num_plot = min(5, len(all_preds))
fig, axes = plt.subplots(1, num_plot, figsize=(num_plot*3, 4))
if num_plot == 1:
    axes = [axes]

for i in range(num_plot):
    axes[i].bar(range(OUTPUT_DIM), all_targets[i], alpha=0.5, label='True', color='blue')
    axes[i].bar(range(OUTPUT_DIM), all_preds[i], alpha=0.5, label='Pred', color='red')
    axes[i].set_xticks(range(OUTPUT_DIM))
    axes[i].set_xticklabels(['Valence','Arousal','Dom','Lik'])
    axes[i].set_ylim(0,1)
    axes[i].set_title(f"Sample {i} Prediction vs True")
    axes[i].legend()

plt.tight_layout()
plt.show()

# ----------------------------
# 8️⃣ 可视化 Graph 结构 (第一个样本)
# ----------------------------
pic_dir = os.path.join(os.path.dirname(__file__), "..", "pictures")
os.makedirs(pic_dir, exist_ok=True)

first_batch = next(iter(test_loader))
first_batch = first_batch.to(device)

model.eval()
with torch.no_grad():
    out = model(first_batch)

# 获取第一个样本的图结构
# 获取第一个样本的图结构
mask = (first_batch.batch == 0)
node_features = first_batch.x[mask].cpu().numpy()

# 正确获取第一个样本的边：找到两个节点都在mask中的边
nodes_indices = torch.nonzero(mask, as_tuple=True)[0]  # 获取属于第一个样本的节点索引
# 找到边的两个端点都属于第一个样本的边
edge_mask = (first_batch.batch[first_batch.edge_index[0]] == 0) & (first_batch.batch[first_batch.edge_index[1]] == 0)
edge_index_full = first_batch.edge_index[:, edge_mask].cpu().numpy()

# 将全局节点索引转换为子图节点索引
node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(nodes_indices)}
mapped_edges = []
for i in range(edge_index_full.shape[1]):
    src, dst = edge_index_full[0, i], edge_index_full[1, i]
    mapped_src, mapped_dst = node_mapping[src], node_mapping[dst]
    mapped_edges.append((mapped_src, mapped_dst))

G = nx.Graph()
G.add_edges_from(mapped_edges)

plt.figure(figsize=(6,6))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
plt.title("Graph Structure (First Sample in Batch)")
save_path = os.path.join(pic_dir, f"{MODEL_TYPE}_graph_structure_first_sample.png")
plt.savefig(save_path)
plt.close()
print(f"Graph structure saved to {save_path}")
print(f"Number of nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
print(f"Node feature shape: {node_features.shape}")

# ----------------------------
# 9️⃣ 计算相关系数评估
# ----------------------------
correlations = []
for dim in range(OUTPUT_DIM):
    corr = np.corrcoef(all_targets[:, dim], all_preds[:, dim])[0, 1]
    correlations.append(corr)
    print(f"Dimension {dim} ({['Valence','Arousal','Dom','Lik'][dim]}) Correlation: {corr:.4f}")

avg_corr = np.mean(correlations)
print(f"Average Correlation: {avg_corr:.4f}")