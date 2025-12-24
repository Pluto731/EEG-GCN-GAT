# src/train.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

# ----------------------------
# 0️⃣ 路径与配置设置
# ----------------------------
# 动态添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import *
from src.models.gcn import GCN
from src.models.gat import GAT

# ----------------------------
# 辅助函数：保存/加载检查点
# ----------------------------
def save_checkpoint(model, optimizer, epoch, train_losses, train_mae, val_losses, val_mae, best_val_loss, patience_counter, save_path):
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
    checkpoint_dir = os.path.join(SAVE_DIR, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{MODEL_TYPE}_checkpoint_epoch_")]
    if not checkpoints:
        return None
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    return os.path.join(checkpoint_dir, f"{MODEL_TYPE}_checkpoint_epoch_{latest_epoch}.pt")

# ----------------------------
# 1️⃣ 加载预处理数据
# ----------------------------
processed_file_name = f'deap_{GRAPH_TYPE}_window1s.pt'
processed_file_path = os.path.join(DATA_PROCESSED_DIR, processed_file_name)

print("="*50)
print(f"Loading processed data from: {processed_file_path}")
if not os.path.exists(processed_file_path):
    raise FileNotFoundError(f"Processed file not found! Please run 'src/preprocess.py' first.")

# 加载数据列表
data_list = torch.load(processed_file_path)
print(f"Successfully loaded {len(data_list)} samples.")

# ----------------------------
# 2️⃣ 数据预处理 (标签转换 & 特征归一化)
# ----------------------------
print("Preprocessing features and labels...")

# A. 提取所有特征用于计算 Mean/Std
all_features = torch.cat([data.x for data in data_list], dim=0).numpy()
scaler = StandardScaler()
scaler.fit(all_features)
print(f"Feature statistics - Mean: {scaler.mean_[:3]}, Std: {scaler.scale_[:3]}")

# B. 遍历数据进行转换
processed_list = []
for data in tqdm(data_list, desc="Normalizing"):
    # 1. 特征归一化
    data.x = torch.tensor(scaler.transform(data.x.numpy()), dtype=torch.float)
    
    # 2. 标签处理
    # 原始标签是 [1, 4] 形状，值域 1-9
    labels = data.y
    if LABEL_TYPE == 'binary':
        # 二分类：大于阈值设为 1.0，否则 0.0
        binary_labels = (labels >= LABEL_THRESHOLD).float()
        data.y = binary_labels
    else:
        # 回归：归一化到 0-1 之间 (可选，这里先保留原始 1-9 或做简单的 min-max)
        # data.y = (labels - 1) / 8.0 
        pass 
        
    processed_list.append(data)

# ----------------------------
# 3️⃣ 数据集划分
# ----------------------------
# 随机划分 (70% Train, 15% Val, 15% Test)
train_data, temp_data = train_test_split(processed_list, test_size=0.3, random_state=42, shuffle=True)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Dataset split: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")

# 构建 DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 4️⃣ 模型初始化
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if MODEL_TYPE.lower() == "gcn":
    model = GCN(in_channels=NODE_FEATURE_DIM, hidden_channels=HIDDEN_DIM, out_channels=OUTPUT_DIM, dropout=DROPOUT)
elif MODEL_TYPE.lower() == "gat":
    model = GAT(in_channels=NODE_FEATURE_DIM, hidden_channels=HIDDEN_DIM, out_channels=OUTPUT_DIM, heads=HEADS, dropout=DROPOUT)
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

model = model.to(device)

# 优化器与损失函数
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# 如果是二分类且输出是 logits (无sigmoid)，使用 BCEWithLogitsLoss
# 如果要做回归，使用 MSELoss
if LABEL_TYPE == 'binary':
    criterion = nn.BCEWithLogitsLoss() 
    print("Loss Function: BCEWithLogitsLoss (Binary Classification)")
else:
    criterion = nn.MSELoss()
    print("Loss Function: MSELoss (Regression)")

# ----------------------------
# 5️⃣ 训练准备 (断点续训)
# ----------------------------
start_epoch = 0
train_losses, train_mae, val_losses, val_mae = [], [], [], []
best_val_loss = float("inf")
patience_counter = 0

if RESUME_TRAINING:
    latest_checkpoint_path = get_latest_checkpoint()
    if latest_checkpoint_path:
        print(f"Resuming from {latest_checkpoint_path}")
        start_epoch, train_losses, train_mae, val_losses, val_mae, best_val_loss, patience_counter = \
            load_checkpoint(model, optimizer, latest_checkpoint_path)
    else:
        print("No checkpoint found, starting fresh.")

checkpoint_dir = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
lr_history = []

# ----------------------------
# 6️⃣ 训练循环
# ----------------------------
print(f"Start training for {EPOCHS} epochs...")

for epoch in range(start_epoch + 1, EPOCHS + 1):
    # --- Train ---
    model.train()
    epoch_loss = 0
    epoch_mae = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)  # shape: (batch_size, 4)
        y_true = batch.y.view(out.size(0), -1) # shape: (batch_size, 4)
        
        loss = criterion(out, y_true)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch.num_graphs
        # 计算 MAE 仅作参考 (如果是二分类，这里的 MAE 是 logit 和 0/1 的差，意义不大，但保留用于监测收敛)
        epoch_mae += torch.mean(torch.abs(torch.sigmoid(out) - y_true)).item() * batch.num_graphs if LABEL_TYPE == 'binary' \
                     else torch.mean(torch.abs(out - y_true)).item() * batch.num_graphs

    epoch_loss /= len(train_loader.dataset)
    epoch_mae /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_mae.append(epoch_mae)

    # --- Validation ---
    model.eval()
    val_epoch_loss = 0
    val_epoch_mae = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            y_true = batch.y.view(out.size(0), -1)
            loss = criterion(out, y_true)
            
            val_epoch_loss += loss.item() * batch.num_graphs
            val_epoch_mae += torch.mean(torch.abs(torch.sigmoid(out) - y_true)).item() * batch.num_graphs if LABEL_TYPE == 'binary' \
                             else torch.mean(torch.abs(out - y_true)).item() * batch.num_graphs

    val_epoch_loss /= len(val_loader.dataset)
    val_epoch_mae /= len(val_loader.dataset)
    val_losses.append(val_epoch_loss)
    val_mae.append(val_epoch_mae)

    scheduler.step(val_epoch_loss)
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)

    print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | LR: {current_lr:.6f}")

    # --- Save Best & Checkpoint ---
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{MODEL_TYPE}_best.pt"))
    else:
        patience_counter += 1

    if epoch % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(model, optimizer, epoch, train_losses, train_mae, val_losses, val_mae, best_val_loss, patience_counter, 
                        os.path.join(checkpoint_dir, f"{MODEL_TYPE}_checkpoint_epoch_{epoch}.pt"))

    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# ----------------------------
# 7️⃣ 绘图 (Loss & LR)
# ----------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title(f'{MODEL_TYPE} Loss Curve')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(lr_history, color='g')
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 8️⃣ 测试集评估
# ----------------------------
print("\n=== Evaluating on Test Set ===")
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{MODEL_TYPE}_best.pt")))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(device)
        out = model(batch)
        y_true = batch.y.view(out.size(0), -1)
        
        if LABEL_TYPE == 'binary':
            # 将 Logits 转为 概率 -> 0/1
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
        else:
            preds = out
            
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_true.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# ----------------------------
# 9️⃣ 计算详细指标
# ----------------------------
results = {'model': MODEL_TYPE}
dims = ['Valence', 'Arousal', 'Dominance', 'Liking']

if LABEL_TYPE == 'binary':
    print(f"{'Dimension':<12} | {'Acc':<8} | {'F1-Score':<8}")
    print("-" * 35)
    total_acc = 0
    for i, dim_name in enumerate(dims):
        acc = accuracy_score(all_targets[:, i], all_preds[:, i])
        f1 = f1_score(all_targets[:, i], all_preds[:, i], average='macro')
        print(f"{dim_name:<12} | {acc:.4f}   | {f1:.4f}")
        results[f'{dim_name}_acc'] = acc
        total_acc += acc
    print("-" * 35)
    print(f"Average Acc: {total_acc/4:.4f}")
    results['avg_acc'] = total_acc/4

else:
    print(f"{'Dimension':<12} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8}")
    print("-" * 45)
    for i, dim_name in enumerate(dims):
        rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        print(f"{dim_name:<12} | {rmse:.4f}   | {mae:.4f}   | {r2:.4f}")

# 保存结果
with open(os.path.join(SAVE_DIR, f"{MODEL_TYPE}_test_results.json"), 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nDone! Results saved to {SAVE_DIR}")