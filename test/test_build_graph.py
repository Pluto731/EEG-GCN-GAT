import sys
import os
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.build_graph import build_graph, EEGGraphDataset

# 模拟 segment
np.random.seed(0)
fake_segment = np.random.randn(32, 256)
fake_label = np.array([1, 0, 1, 0])  # DEAP 四个维度情绪

# 1️⃣ 测试单个图
graph = build_graph(fake_segment, fake_label, graph_type='plv', thresh=0.5)
print("Graph x shape:", graph.x.shape)
print("Graph edge_index shape:", graph.edge_index.shape)
print("Graph y:", graph.y)

# 2️⃣ 测试 dataset
segments = np.stack([fake_segment]*10)  # 10 个 segment
labels = np.stack([fake_label]*10)
dataset = EEGGraphDataset(segments, labels, graph_type='plv', thresh=0.5)
print("Dataset length:", len(dataset))
print("First graph x shape:", dataset[0].x.shape)
