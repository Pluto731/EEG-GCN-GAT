import sys, os
import torch
import numpy as np
from torch_geometric.data import Data, Batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.gcn import GCN
from src.models.gat import GAT

# 模拟图
num_nodes = 32
x = torch.randn(num_nodes, 5)
edge_index = torch.tensor([[i for i in range(num_nodes)],
                           [i for i in range(num_nodes)]], dtype=torch.long)
y = torch.tensor([[1.,0.,1.,0.]], dtype=torch.float)
batch = torch.zeros(num_nodes, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y, batch=batch)

# 批量（1图）
batch_data = Batch.from_data_list([data])

# 测试 GCN
gcn_model = GCN()
out = gcn_model(batch_data)
print("GCN output:", out)

# 测试 GAT
gat_model = GAT()
out = gat_model(batch_data)
print("GAT output:", out)
