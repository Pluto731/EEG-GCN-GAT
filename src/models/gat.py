"""
2 层 GATConv + ELU + Dropout
第 1 层多头注意力，输出 concat
graph-level pooling + Linear 输出 4 维
"""
# src/models/gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=4, heads=4, dropout=0.5):
        """
        in_channels: 节点特征维度
        hidden_channels: GAT 隐藏维度
        out_channels: 输出维度 (情绪标签维度)
        heads: multi-head attention 数量
        dropout: dropout rate
        """
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=1, concat=True, dropout=dropout)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # graph-level pooling
        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x
