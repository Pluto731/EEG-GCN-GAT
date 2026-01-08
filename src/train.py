# src/train.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json

# ----------------------------
# 0️⃣ 路径与配置设置
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import *
from src.models.gcn import GCN
from src.models.gat import GAT

# ----------------------------
# 1️⃣ 加载预处理数据
# ----------------------------
processed_file_name = f'deap_{GRAPH_TYPE}_window1s.pt'
processed_file_path = os.path.join(DATA_PROCESSED_DIR, processed_file_name)

print("="*50)
print(f"Loading processed data from: {processed_file_path}")
if not os.path.exists(processed_file_path):
    raise FileNotFoundError(f"Processed file not found! Please run 'src/preprocess.py' first.")

data_list = torch.load(processed_file_path)
print(f"Successfully loaded {len(data_list)} samples.")

# ----------------------------
# [新增] 2.5 特征平滑 (Feature Smoothing)
# ----------------------------
print("Applying Moving Average Smoothing (Window Size=5)...")

def moving_average_smoothing(data_list, window_size=5):
    # 按 Subject ID 分组处理
    subject_ids = torch.tensor([d.subject_id.item() for d in data_list])
    unique_subs = torch.unique(subject_ids).tolist()
    
    smoothed_data_list = []
    
    for sub in unique_subs:
        # 找出该受试者的所有样本
        indices = (subject_ids == sub).nonzero(as_tuple=True)[0]
        sub_samples = [data_list[i] for i in indices]
        
        # 提取特征矩阵: (Time, Nodes, Feats)
        x_stack = torch.stack([d.x for d in sub_samples]) # (T, 32, 5)
        
        # 对时间维度 (dim=0) 进行平滑
        # 使用 numpy 的 convolve 实现，简单高效
        x_numpy = x_stack.numpy()
        x_smoothed = np.zeros_like(x_numpy)
        
        for node in range(x_numpy.shape[1]):
            for band in range(x_numpy.shape[2]):
                signal = x_numpy[:, node, band]
                # 简单移动平均卷积核
                kernel = np.ones(window_size) / window_size
                # same 模式保持长度不变
                smooth_sig = np.convolve(signal, kernel, mode='same')
                x_smoothed[:, node, band] = smooth_sig
                
        # 将平滑后的特征写回 Data 对象
        for i, idx in enumerate(indices):
            data_list[idx].x = torch.tensor(x_smoothed[i], dtype=torch.float)
            
    return data_list

# 执行平滑
processed_list = moving_average_smoothing(data_list, window_size=5)
print("Smoothing done.")

# ----------------------------
# 2️⃣ 全局数据预处理 (特征归一化 & 标签二值化)
# ----------------------------
print("Preprocessing features globally...")

# A. 全局计算 Mean/Std (为了保持尺度一致)
all_features = torch.cat([data.x for data in data_list], dim=0).numpy()
scaler = StandardScaler()
scaler.fit(all_features)

processed_list = []
for data in tqdm(data_list, desc="Normalizing"):
    # 特征归一化
    data.x = torch.tensor(scaler.transform(data.x.numpy()), dtype=torch.float)
    
    # 标签二值化处理
    if LABEL_TYPE == 'binary':
        binary_labels = (data.y >= LABEL_THRESHOLD).float()
        data.y = binary_labels
    
    processed_list.append(data)

# ----------------------------
# 3️⃣ Subject-Dependent 训练主循环
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 获取所有受试者 ID
subject_ids = torch.tensor([d.subject_id.item() for d in processed_list])
unique_subjects = torch.unique(subject_ids).tolist()
# 如果只想测试前几个人调试代码，可以解开下面这行
# unique_subjects = unique_subjects[:3] 

print(f"\n🚀 Starting Subject-Dependent Experiments for {len(unique_subjects)} subjects...")

# 存储每个人的结果
final_results = {
    'valence_acc': [], 'arousal_acc': [], 'dominance_acc': [], 'liking_acc': [],
    'avg_acc': []
}

# === 大循环：遍历每个受试者 ===
for sub_id in unique_subjects:
    print(f"\n{'='*20} Training Subject {sub_id} {'='*20}")
    
    # 3.1 筛选当前受试者数据
    sub_data = [d for d in processed_list if d.subject_id.item() == sub_id]
    
    # 3.2 划分训练/测试集，并增加验证集用于更准确的早停
    train_data, temp_data = train_test_split(sub_data, test_size=0.3, random_state=42, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, shuffle=True)

    # 构建 Loader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3.3 ✅ 计算类别权重 (解决 F1 低的问题)
    # 统计训练集中每一列正样本的数量
    y_train_all = torch.stack([d.y for d in train_data]) # (N, 4)
    num_pos = torch.sum(y_train_all, dim=0) # (4,)
    num_neg = len(train_data) - num_pos
    # pos_weight = negative / positive
    pos_weight = (num_neg / (num_pos + 1e-5)).to(device)
    
    print(f"  > Class Balance (Pos/Total): {num_pos.cpu().numpy()} / {len(train_data)}")
    print(f"  > Calculated Pos Weight: {pos_weight.cpu().numpy()}")

    # 3.4 初始化新模型 (每个人一个新脑子)
    # 使用配置文件中的参数，但根据实际情况进行微调
    if MODEL_TYPE.lower() == "gcn":
        model = GCN(in_channels=NODE_FEATURE_DIM, 
                    hidden_channels=HIDDEN_DIM,  # 使用配置文件中的参数
                    out_channels=OUTPUT_DIM, 
                    dropout=DROPOUT)             # 使用配置文件中的参数
    elif MODEL_TYPE.lower() == "gat":
        model = GAT(in_channels=NODE_FEATURE_DIM, 
                    hidden_channels=HIDDEN_DIM,  # 使用配置文件中的参数
                    out_channels=OUTPUT_DIM, 
                    heads=HEADS,                 # 使用配置文件中的参数
                    dropout=DROPOUT)             # 使用配置文件中的参数
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 使用Focal Loss替代BCEWithLogitsLoss以更好地处理类别不平衡
    class FocalLoss(nn.Module):  # 确保继承nn.Module
        def __init__(self, alpha=1, gamma=2, reduce=True):
            super(FocalLoss, self).__init__()  # 添加正确的super()调用
            self.alpha = alpha
            self.gamma = gamma
            self.reduce = reduce

        def forward(self, inputs, targets):
            bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            if self.reduce:
                return torch.mean(focal_loss)
            else:
                return focal_loss

    criterion = FocalLoss(alpha=1, gamma=2)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 3.5 内部训练循环
    best_val_acc = 0.0
    patience_counter = 0
    # 为了节省时间，Subject-Dependent 可以在 Epoch 上稍微减少，或者早停更严格
    MAX_PATIENCE_SUB = PATIENCE  # 使用配置文件中的耐心值
    
    # 使用 tqdm 显示进度，但只显示 epoch 简略信息避免刷屏
    pbar = tqdm(range(EPOCHS*2), desc=f"Sub {sub_id}", leave=False)  # 增加训练轮数

    for epoch in pbar:
        # --- Train ---
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 数据增强：对输入特征添加轻微噪声
            noisy_batch = batch.clone()
            noise = torch.randn_like(noisy_batch.x) * 0.01  # 添加小幅度噪声
            noisy_batch.x = noisy_batch.x + noise
            
            out = model(noisy_batch)
            loss = criterion(out, batch.y.view(out.size(0), -1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # --- Validation (作为 Validation) ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:  # 使用验证集进行验证
                batch = batch.to(device)
                out = model(batch)
                # Sigmoid + Threshold
                preds = (torch.sigmoid(out) > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(batch.y.view(out.size(0), -1).cpu())
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # 计算当前平均准确率
        curr_acc = accuracy_score(all_targets.flatten(), all_preds.flatten())
        
        # 更新学习率调度器
        scheduler.step(curr_acc)
        
        # 更新进度条
        pbar.set_postfix({'Loss': f'{epoch_loss/len(train_loader):.4f}', 'Val Acc': f'{curr_acc:.4f}'})
        sys.stdout.flush()  # 确保输出被刷新

        # 早停检查 (基于验证集 Acc)
        if curr_acc > best_val_acc:
            best_val_acc = curr_acc
            patience_counter = 0
            # 保存该 Subject 的最佳模型参数 (临时变量)
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE_SUB:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 3.6 训练结束，使用最佳模型评估最终结果
    model.load_state_dict(best_model_state)
    model.eval()
    
    # 再次在测试集上跑一遍获取详细指标
    # (这部分代码与上面验证类似，为了清晰再写一次)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            # 使用软阈值而非硬阈值，可能提高性能
            probs = torch.sigmoid(out)
            # 可以尝试不同的阈值而非固定的0.5
            threshold = 0.5
            preds = (probs > threshold).float()
            all_preds.append(preds.cpu())
            all_targets.append(batch.y.view(out.size(0), -1).cpu())
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # 记录该 Subject 的四个维度准确率
    dims = ['valence', 'arousal', 'dominance', 'liking']
    sub_accs = []
    print(f"  [Result Sub {sub_id}]", end=" ")
    for i, dim in enumerate(dims):
        acc = accuracy_score(all_targets[:, i], all_preds[:, i])
        f1 = f1_score(all_targets[:, i], all_preds[:, i], average='macro')
        final_results[f'{dim}_acc'].append(acc)
        sub_accs.append(acc)
        print(f"{dim}: {acc:.2f} (F1:{f1:.2f}) |", end=" ")
    
    avg_sub_acc = np.mean(sub_accs)
    final_results['avg_acc'].append(avg_sub_acc)
    print(f"Avg: {avg_sub_acc:.4f}")

# ----------------------------
# 4️⃣ 汇总输出最终结果
# ----------------------------
print("\n" + "="*50)
print("FINAL SUBJECT-DEPENDENT RESULTS (Average over 32 subjects)")
print("="*50)

mean_valence = np.mean(final_results['valence_acc'])
mean_arousal = np.mean(final_results['arousal_acc'])
mean_dominance = np.mean(final_results['dominance_acc'])
mean_liking = np.mean(final_results['liking_acc'])
total_avg = np.mean(final_results['avg_acc'])

print(f"Valence Accuracy : {mean_valence:.4f}")
print(f"Arousal Accuracy : {mean_arousal:.4f}")
print(f"Dominance Accuracy: {mean_dominance:.4f}")
print(f"Liking Accuracy  : {mean_liking:.4f}")
print("-" * 30)
print(f"OVERALL AVERAGE  : {total_avg:.4f}")

# 保存结果到 json
save_path = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_subject_dependent_results_plv.json")
with open(save_path, 'w') as f:
    # 将 numpy 类型转为 float 存 json
    json_results = {k: [float(v) for v in vals] for k, vals in final_results.items()}
    json_results['summary'] = {
        'valence': mean_valence, 'arousal': mean_arousal,
        'dominance': mean_dominance, 'liking': mean_liking,
        'overall': total_avg
    }
    json.dump(json_results, f, indent=4)

print(f"\nDetailed results saved to {save_path}")