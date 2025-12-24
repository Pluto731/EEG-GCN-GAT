# src/models/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

"""
GCN 是图神经网络的基石。它的核心思想是**聚合邻居信息**
即每个电极（节点）的特征会根据邻接矩阵（大脑连接）与其邻居电极的特征进行加权平均
从而更新自己的状态。
"""
class GCN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=4, dropout=0.3):
        """
        in_channels: 节点特征维度 (DE features)
        hidden_channels: GCN 隐藏维度
        out_channels: 输出维度 (情绪标签维度)
        dropout: dropout rate
        """
        super(GCN, self).__init__()
        # 增加网络深度，添加批归一化
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels//2)
        self.bn3 = BatchNorm(hidden_channels//2)
        self.lin1 = nn.Linear(hidden_channels//2, hidden_channels//4)
        self.lin2 = nn.Linear(hidden_channels//4, out_channels)
        self.dropout = dropout

    def forward(self, data):
        """
        data: PyG Data
        data.x -> 节点特征, data.edge_index -> 边索引, data.batch -> batch索引
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # graph-level pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)  # [num_graphs, out_channels]
        return x