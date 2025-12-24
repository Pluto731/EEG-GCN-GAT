# src/models/gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

"""
GAT 是 GCN 的升级版。它的核心思想是**注意力机制**
即每个电极在聚合邻居信息时会智能地判断哪个邻居更重要，并赋予不同的权重。
"""

class GAT(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=4, heads=4, dropout=0.3):
        """
        in_channels: 节点特征维度
        hidden_channels: GAT 隐藏维度
        out_channels: 输出维度 (情绪标签维度)
        heads: multi-head attention 数量
        dropout: dropout rate
        """
        super(GAT, self).__init__()
        # 增加网络深度，添加批归一化
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=heads//2, concat=True, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels * (heads//2))
        self.conv3 = GATConv(hidden_channels*(heads//2), hidden_channels//2, heads=1, concat=True, dropout=dropout)
        self.bn3 = BatchNorm(hidden_channels//2)
        self.lin1 = nn.Linear(hidden_channels//2, hidden_channels//4)
        self.lin2 = nn.Linear(hidden_channels//4, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        # graph-level pooling
        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x