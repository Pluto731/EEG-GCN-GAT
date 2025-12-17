# src/graph/build_graph.py

import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from src.feature.de import extract_de_features
from src.feature.connectivity import pearson_connectivity, plv_connectivity, threshold_matrix

# -----------------------------
# 1️⃣ 辅助函数：根据邻接矩阵生成 edge_index
# -----------------------------
def adjacency_to_edge_index(adj):
    """
    Convert adjacency matrix to PyG edge_index
    Only keep non-zero edges
    """
    row, col = np.nonzero(adj)
    edge_index = np.vstack((row, col))
    return torch.tensor(edge_index, dtype=torch.long)


# -----------------------------
# 2️⃣ 单个 segment 构建 PyG Data
# -----------------------------
def build_graph(segment, label, graph_type='pearson', thresh=0.5):
    """
    segment: np.ndarray, shape (32, T)
    label: int or np.ndarray
    graph_type: 'pearson' / 'plv'
    thresh: float, threshold for adjacency matrix
    """
    # 1️⃣ 节点特征
    x = extract_de_features(segment)           # shape: (32, 5)
    x = torch.tensor(x, dtype=torch.float)

    # 2️⃣ 邻接矩阵
    if graph_type == 'pearson':
        adj = pearson_connectivity(segment)
    elif graph_type == 'plv':
        adj = plv_connectivity(segment)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    # 3️⃣ 可选 threshold
    adj = threshold_matrix(adj, thresh)

    # 4️⃣ edge_index
    edge_index = adjacency_to_edge_index(adj)

    # 5️⃣ label
    y = torch.tensor(label, dtype=torch.float).unsqueeze(0) if np.isscalar(label) else torch.tensor(label, dtype=torch.float)

    # 6️⃣ 构建 PyG Data
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


# -----------------------------
# 3️⃣ 构建整个数据集（InMemoryDataset 可选）
# -----------------------------
class EEGGraphDataset(InMemoryDataset):
    def __init__(self, segments, labels, graph_type='pearson', thresh=0.5, transform=None):
        """
        segments: np.ndarray, shape (N_segments, 32, T)
        labels: np.ndarray, shape (N_segments, label_dim)
        """
        super().__init__(None, transform)
        self.graph_type = graph_type
        self.thresh = thresh

        self.data_list = []
        for i in range(len(segments)):
            seg = segments[i]
            lab = labels[i]
            graph = build_graph(seg, lab, graph_type=graph_type, thresh=thresh)
            self.data_list.append(graph)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
