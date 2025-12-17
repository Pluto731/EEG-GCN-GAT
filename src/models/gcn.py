"""
2 层 GCNConv + ReLU + Dropout
全局平均池化得到 graph-level 表示
输出 4 维（DEAP 四维情绪）
"""
# src/models/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=4, dropout=0.5):
        """
        in_channels: 节点特征维度 (DE features)
        hidden_channels: GCN 隐藏维度
        out_channels: 输出维度 (情绪标签维度)
        dropout: dropout rate
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        """
        data: PyG Data
        data.x -> 节点特征, data.edge_index -> 边索引, data.batch -> batch索引
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # graph-level pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        x = self.lin(x)  # [num_graphs, out_channels]
        return x
