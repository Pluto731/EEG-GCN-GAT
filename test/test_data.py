import torch
import os
import sys
import numpy as np
from collections import Counter

# -----------------------------------------------------------
# 1. 路径设置 (自动引用 config 中的路径)
# -----------------------------------------------------------
# 将 src 加入路径，以便导入 config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config import DATA_PROCESSED_DIR, GRAPH_TYPE, LABEL_THRESHOLD
except ImportError:
    # 如果找不到 config，提供后备默认值
    print("Warning: Could not import config. Using default paths.")
    DATA_PROCESSED_DIR = "./data/processed"
    GRAPH_TYPE = "pearson"
    LABEL_THRESHOLD = 5.0

def inspect_data():
    # 构建文件名 (需与 preprocess.py 中保存的命名一致)
    file_name = f'deap_{GRAPH_TYPE}_window1s.pt'
    file_path = os.path.join(DATA_PROCESSED_DIR, file_name)

    print("=" * 60)
    print(f"🔍 INSPECTING DATA: {file_path}")
    print("=" * 60)

    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        print("Please run 'src/preprocess.py' first.")
        return

    # 2. 加载数据
    try:
        data_list = torch.load(file_path)
        print(f"✅ File loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # 3. 数据集整体统计
    num_samples = len(data_list)
    print(f"\n📊 DATASET STATISTICS:")
    print(f"  - Total Samples: {num_samples}")
    # 预期数量估算: 32人 * 40试验 * 60秒 = 76,800
    print(f"  - Expected ~76,800 samples? {'Yes' if 76000 < num_samples < 77000 else 'No (Check preprocess logic)'}")

    if num_samples == 0:
        print("❌ Dataset is empty!")
        return

    # 4. 抽取第 1 个样本进行详细检查
    sample = data_list[0]
    print(f"\n🔬 SAMPLE STRUCTURE (Sample 0):")
    print(f"  - Data Object: {sample}")
    
    # 检查节点特征 x
    # 预期: (32, 5) -> (Channels, Bands)
    print(f"  - Node Features (x): {sample.x.shape}  [Expected: (32, 5)]")
    print(f"    - Min val: {sample.x.min():.4f}, Max val: {sample.x.max():.4f}")
    if torch.isnan(sample.x).any():
        print("    ⚠️ WARNING: NaN values found in features!")

    # 检查边 edge_index
    # 预期: (2, Num_Edges)
    print(f"  - Edge Index:        {sample.edge_index.shape}")
    print(f"    - Num Edges: {sample.edge_index.shape[1]}")
    
    # 检查标签 y
    # 预期: (1, 4) -> (Valence, Arousal, Dominance, Liking)
    print(f"  - Labels (y):        {sample.y.shape} -> Values: {sample.y.tolist()}")
    
    # 检查 Subject ID
    if hasattr(sample, 'subject_id'):
        print(f"  - Subject ID:        {sample.subject_id.item()}")

    # 5. 检查标签分布 (模拟二分类)
    # 抽样检查前 1000 个样本的 Valence 和 Arousal
    print(f"\n📈 LABEL DISTRIBUTION (First 1000 samples):")
    print(f"  Threshold: {LABEL_THRESHOLD}")
    
    subset = data_list[:1000]
    valence_labels = [1 if d.y[0, 0] >= LABEL_THRESHOLD else 0 for d in subset]
    arousal_labels = [1 if d.y[0, 1] >= LABEL_THRESHOLD else 0 for d in subset]

    print(f"  - Valence (High/Low): {Counter(valence_labels)}")
    print(f"  - Arousal (High/Low): {Counter(arousal_labels)}")

    print("\n✅ Data verification complete. Ready for training.")
    print("=" * 60)

if __name__ == "__main__":
    inspect_data()