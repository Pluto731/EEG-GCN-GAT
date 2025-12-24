# preprocess.py
import os
import torch
import numpy as np
from tqdm import tqdm  # 进度条库，建议 pip install tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 引入你的模块
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, FS, WINDOW_SIZE, STEP_SIZE, BASELINE_LEN, GRAPH_TYPE, THRESHOLD
from src.data.load_deap import DEAPLoader
from src.feature.de import extract_de_features
from src.feature.connectivity import pearson_connectivity, plv_connectivity, threshold_matrix
# 注意：你需要根据 build_graph.py 里的逻辑稍作调整，这里我们将直接手动组装以提升批量处理效率
from torch_geometric.data import Data

def process_and_save():
    # 1. 准备目录
    if not os.path.exists(DATA_PROCESSED_DIR):
        os.makedirs(DATA_PROCESSED_DIR)
        print(f"Created directory: {DATA_PROCESSED_DIR}")

    # 2. 加载原始数据
    print("Loading raw DEAP data...")
    loader = DEAPLoader(data_dir=DATA_RAW_DIR)
    # data: (1280, 32, 8064), labels: (1280, 4), subjects: (1280,)
    all_data, all_labels, all_subjects = loader.load_all_subjects()
    
    processed_data_list = []
    
    print(f"Start processing... (Graph Type: {GRAPH_TYPE})")
    print(f"Total trials: {len(all_data)}")
    
    # 3. 循环处理每个 Trial
    # 使用 tqdm 显示进度条，因为处理会比较慢
    for i in tqdm(range(len(all_data)), desc="Processing Trials"):
        raw_trial = all_data[i]  # shape (32, 8064)
        label_trial = all_labels[i] # shape (4,)
        subject_id = all_subjects[i]
        
        # --- 切片策略 (Sliding Window) ---
        # DEAP 数据前 3秒是基线，从第 3秒开始取 60秒数据
        # raw_trial.shape[1] = 8064
        n_samples = raw_trial.shape[1]
        
        start_idx = BASELINE_LEN # 从第 384 点开始
        
        while start_idx + WINDOW_SIZE <= n_samples:
            # A. 截取 1秒 片段
            segment = raw_trial[:, start_idx : start_idx + WINDOW_SIZE] # (32, 128)
            
            # B. 提取 DE 特征 (Node Features)
            # extract_de_features 返回 (32, 5)
            # 确保你的 de.py 里使用了 log 变换后的 DE
            x = extract_de_features(segment, fs=FS)
            x_tensor = torch.tensor(x, dtype=torch.float)
            
            # C. 构建邻接矩阵 (Edge Index)
            if GRAPH_TYPE == 'pearson':
                adj = pearson_connectivity(segment)
            elif GRAPH_TYPE == 'plv':
                adj = plv_connectivity(segment)
            
            # 阈值处理
            adj = threshold_matrix(adj, thresh=THRESHOLD)
            
            # 转换为 PyG 的 edge_index
            row, col = np.nonzero(adj)
            edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
            
            # 也可以选择保存边权重
            edge_attr = torch.tensor(adj[row, col], dtype=torch.float)

            # D. 处理标签 (Labels)
            # 这里我们保存所有原始标签，训练时再做二分类（比如 label > 5 set 1 else 0）
            y_tensor = torch.tensor(label_trial, dtype=torch.float).unsqueeze(0) 
            
            # E. 封装为 PyG Data 对象
            # 我们额外保存 subject_id，方便做 Leave-One-Subject-Out 交叉验证
            data = Data(x=x_tensor, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr,
                        y=y_tensor,
                        subject_id=torch.tensor([subject_id]))
            
            processed_data_list.append(data)
            
            # 移动窗口
            start_idx += STEP_SIZE

    # 4. 保存到硬盘
    save_path = os.path.join(DATA_PROCESSED_DIR, f'deap_{GRAPH_TYPE}_window1s.pt')
    print(f"Saving {len(processed_data_list)} graphs to {save_path}...")
    torch.save(processed_data_list, save_path)
    print("Done!")

if __name__ == "__main__":
    process_and_save()