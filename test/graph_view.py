# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import networkx as nx
# import os
# import sys
# from scipy.interpolate import griddata

# # ---------------------------------------------------------
# # 1. 配置与 DEAP 通道定义
# # ---------------------------------------------------------
# # DEAP 数据集 32 个 EEG 通道的标准名称
# CHANNEL_NAMES = [
#     'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 
#     'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 
#     'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 
#     'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
# ]

# # 简单的 2D 投影坐标 (用于画脑地形图和网络图)
# # 这是一个近似的 10-20 系统投影
# CHANNEL_POS = {
#     'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9), 'AF3': (-0.4, 0.7), 'AF4': (0.4, 0.7),
#     'F7': (-0.8, 0.5), 'F3': (-0.5, 0.5), 'Fz': (0.0, 0.5), 'F4': (0.5, 0.5), 'F8': (0.8, 0.5),
#     'FC5': (-0.7, 0.25), 'FC1': (-0.3, 0.25), 'FC2': (0.3, 0.25), 'FC6': (0.7, 0.25),
#     'T7': (-0.9, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.5, 0.0), 'T8': (0.9, 0.0),
#     'CP5': (-0.7, -0.25), 'CP1': (-0.3, -0.25), 'CP2': (0.3, -0.25), 'CP6': (0.7, -0.25),
#     'P7': (-0.8, -0.5), 'P3': (-0.4, -0.5), 'Pz': (0.0, -0.5), 'P4': (0.4, -0.5), 'P8': (0.8, -0.5),
#     'PO3': (-0.3, -0.8), 'PO4': (0.3, -0.8), 'O1': (-0.3, -1.0), 'Oz': (0.0, -1.0), 'O2': (0.3, -1.0)
# }

# # 频带名称
# BAND_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# # ---------------------------------------------------------
# # 2. 辅助绘图函数
# # ---------------------------------------------------------
# def plot_head_outline(ax):
#     """画一个简单的头部轮廓"""
#     theta = np.linspace(0, 2 * np.pi, 100)
#     x = np.cos(theta)
#     y = np.sin(theta)
#     ax.plot(x, y, color='black', linewidth=2)
#     # 鼻子
#     ax.plot([0, 0.1, -0.1, 0], [1.0, 1.1, 1.1, 1.0], color='black', linewidth=2)
#     ax.set_aspect('equal')
#     ax.axis('off')

# def visualize_graph_structure(data, title="Brain Connectivity Graph", ax=None):
#     """1. 可视化脑网络拓扑结构"""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 8))
    
#     G = nx.Graph()
#     # 添加节点
#     for i, name in enumerate(CHANNEL_NAMES):
#         if name in CHANNEL_POS:
#             G.add_node(i, pos=CHANNEL_POS[name], label=name)
            
#     # 添加边
#     edge_index = data.edge_index.numpy()
#     edge_attr = data.edge_attr.numpy() if hasattr(data, 'edge_attr') else np.ones(edge_index.shape[1])
    
#     # 为了视觉清晰，只画最强的边 (Top 20%)
#     # 去除自身联系加深颜色
#     edge_index = edge_index[:, edge_index[0] != edge_index[1]]
#     threshold = np.percentile(edge_attr, 80)
#     for i in range(edge_index.shape[1]):
#         u, v = edge_index[0, i], edge_index[1, i]
#         w = edge_attr[i]
#         if w > threshold:
#             G.add_edge(u, v, weight=w)

#     pos = nx.get_node_attributes(G, 'pos')
    
#     # 绘图
#     plot_head_outline(ax)
    
#     # 画边
#     edges = G.edges()
#     weights = [G[u][v]['weight'] for u, v in edges]
#     # 深颜色表示强连接
#     nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.5, edge_color=weights, edge_cmap=plt.cm.Blues)
    
#     # 画点
#     nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='#ffcc99', edgecolors='black')
#     nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
#     ax.set_title(title, fontsize=14)

# def visualize_connectivity_matrix(data, title="Adjacency Matrix", ax=None):
#     """2. 可视化邻接矩阵热力图"""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6))
        
#     adj = np.zeros((32, 32))
#     rows, cols = data.edge_index.numpy()
#     weights = data.edge_attr.numpy() if hasattr(data, 'edge_attr') else np.ones(len(rows))
    
#     adj[rows, cols] = weights
    
#     sns.heatmap(adj, cmap="viridis", square=True, ax=ax, cbar_kws={"shrink": .8})
#     ax.set_xlabel("Channel Index")
#     ax.set_ylabel("Channel Index")
#     ax.set_title(title)

# def visualize_de_features(data, ax=None):
#     """3. 可视化 DE 特征分布 (Boxplot)"""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 6))
    
#     # shape: (32, 5)
#     features = data.x.numpy()
    
#     # 转换为 DataFrame 格式方便绘图
#     data_points = []
#     for band_idx in range(5):
#         vals = features[:, band_idx]
#         for v in vals:
#             data_points.append({'Band': BAND_NAMES[band_idx], 'DE Value': v})
            
#     import pandas as pd
#     df = pd.DataFrame(data_points)
    
#     sns.boxplot(x='Band', y='DE Value', data=df, ax=ax, palette="Set2")
#     ax.set_title("DE Feature Distribution (Single Sample)")
#     ax.grid(True, alpha=0.3)

# def visualize_topomap(data, band_idx=2, title="Alpha Band Energy", ax=None):
#     """4. 模拟脑地形图 (插值热力图)"""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 8))
        
#     values = data.x[:, band_idx].numpy() # 取某个频带的能量
    
#     # 获取坐标
#     coords = []
#     for name in CHANNEL_NAMES:
#         coords.append(CHANNEL_POS.get(name, (0, 0)))
#     coords = np.array(coords)
    
#     # 创建网格
#     grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
    
#     # 插值
#     grid_z = griddata(coords, values, (grid_x, grid_y), method='cubic')
    
#     # 绘图
#     plot_head_outline(ax)
#     contour = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='jet', alpha=0.9, extend='both')
    
#     # 画传感器位置
#     ax.scatter(coords[:, 0], coords[:, 1], c='black', s=20, marker='.')
    
#     # 不要在子图中加 colorbar，容易破坏布局
#     ax.set_title(title)

# def visualize_labels_distribution(data_list, ax=None):
#     """5. 可视化整个数据集的标签分布"""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 5))
        
#     all_labels = []
#     # 随机抽样 1000 个样本，避免太慢
#     sample_indices = np.random.choice(len(data_list), min(1000, len(data_list)), replace=False)
#     for i in sample_indices:
#         all_labels.append(data_list[i].y.numpy().flatten())
        
#     all_labels = np.array(all_labels) # (N, 4)
    
#     import pandas as pd
#     df = pd.DataFrame(all_labels, columns=['Valence', 'Arousal', 'Dominance', 'Liking'])
#     df_melt = df.melt(var_name='Dimension', value_name='Score')
    
#     sns.histplot(data=df_melt, x='Score', hue='Dimension', element="step", stat="density", common_norm=False, ax=ax)
#     ax.set_title("Label Distribution (Sampled)")
#     ax.axvline(x=5.0, color='red', linestyle='--', label='Threshold (5.0)')

# # ---------------------------------------------------------
# # 3. 主程序
# # ---------------------------------------------------------
# def main():
#     # 这里填写你想可视化的文件路径
#     # 优先看 Pearson，因为它是最常用的
#     file_path_pearson = r"E:\Learning\大学课程\计算神经工程\LastWork\data\processed\deap_pearson_window1s.pt"
#     file_path_plv = r"E:\Learning\大学课程\计算神经工程\LastWork\data\processed\deap_plv_window1s.pt"

#     target_file = file_path_plv if os.path.exists(file_path_plv) else file_path_pearson
    
#     print(f"Loading data from: {target_file}")
#     try:
#         data_list = torch.load(target_file)
#     except Exception as e:
#         print(f"Error loading file: {e}")
#         return

#     # 取第一个样本进行微观展示
#     sample = data_list[0]
    
#     # --- 创建画布 ---
#     # 2行3列的布局
#     fig = plt.figure(figsize=(18, 12))
    
#     # 1. 脑网络图 (左上)
#     ax1 = plt.subplot(2, 3, 1)
#     visualize_graph_structure(sample, title="Brain Connectivity (Top 20% Edges)", ax=ax1)
    
#     # 2. 邻接矩阵 (中上)
#     ax2 = plt.subplot(2, 3, 2)
#     visualize_connectivity_matrix(sample, title="Connectivity Matrix", ax=ax2)
    
#     # 3. DE 特征分布 (右上)
#     ax3 = plt.subplot(2, 3, 3)
#     visualize_de_features(sample, ax=ax3)
    
#     # 4. Alpha 波地形图 (左下)
#     ax4 = plt.subplot(2, 3, 4)
#     # Band index 2 is Alpha (Delta, Theta, Alpha, Beta, Gamma)
#     visualize_topomap(sample, band_idx=2, title="Alpha Band Topomap", ax=ax4)
    
#     # 5. Gamma 波地形图 (中下)
#     ax5 = plt.subplot(2, 3, 5)
#     # Band index 4 is Gamma
#     visualize_topomap(sample, band_idx=4, title="Gamma Band Topomap", ax=ax5)
    
#     # 6. 标签分布 (右下)
#     ax6 = plt.subplot(2, 3, 6)
#     visualize_labels_distribution(data_list, ax=ax6)
    
#     plt.suptitle(f"Data Visualization: {os.path.basename(target_file)}", fontsize=16)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # 保存图片
#     save_path = os.path.join(os.path.dirname(target_file), 'visualization_dashboard_PLV.png')
#     plt.savefig(save_path, dpi=300)
#     print(f"Visualization saved to: {save_path}")
#     plt.show()

# if __name__ == "__main__":
#     # 设置样式
#     sns.set_style("whitegrid")
#     main()

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import sys
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# 1. 配置与 DEAP 通道定义
# ---------------------------------------------------------
# DEAP 数据集 32 个 EEG 通道的标准名称
CHANNEL_NAMES = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 
    'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 
    'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]

# 近似 2D 投影坐标 (用于画脑地形图和网络图)
CHANNEL_POS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9), 'AF3': (-0.4, 0.7), 'AF4': (0.4, 0.7),
    'F7': (-0.8, 0.5), 'F3': (-0.5, 0.5), 'Fz': (0.0, 0.5), 'F4': (0.5, 0.5), 'F8': (0.8, 0.5),
    'FC5': (-0.7, 0.25), 'FC1': (-0.3, 0.25), 'FC2': (0.3, 0.25), 'FC6': (0.7, 0.25),
    'T7': (-0.9, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.5, 0.0), 'T8': (0.9, 0.0),
    'CP5': (-0.7, -0.25), 'CP1': (-0.3, -0.25), 'CP2': (0.3, -0.25), 'CP6': (0.7, -0.25),
    'P7': (-0.8, -0.5), 'P3': (-0.4, -0.5), 'Pz': (0.0, -0.5), 'P4': (0.4, -0.5), 'P8': (0.8, -0.5),
    'PO3': (-0.3, -0.8), 'PO4': (0.3, -0.8), 'O1': (-0.3, -1.0), 'Oz': (0.0, -1.0), 'O2': (0.3, -1.0)
}

# 频带名称
BAND_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# ---------------------------------------------------------
# 2. 核心分析功能：寻找最佳阈值
# ---------------------------------------------------------
def analyze_optimal_threshold(data_list):
    """
    计算每个维度的统计特性，推荐最佳二分类阈值
    """
    print("\n" + "="*80)
    print("📊 THRESHOLD ANALYSIS (寻找最佳二分类阈值)")
    print("="*80)
    
    # 提取所有标签 (假设 data.y 存储的是原始 1-9 的分数)
    # 如果已经是 0/1，这里的分析意义不大，但代码兼容
    all_labels = []
    # 为了速度，最多分析前 5000 个样本
    sample_size = min(len(data_list), 5000)
    for i in range(sample_size):
        all_labels.append(data_list[i].y.numpy().flatten())
    all_labels = np.array(all_labels)

    dims = ['Valence', 'Arousal', 'Dominance', 'Liking']
    stats_results = {}
    
    print(f"{'Dimension':<12} | {'Mean':<6} | {'Median':<6} | {'Std':<6} | {'>5.0 Ratio':<12} | {'Recommendation'}")
    print("-" * 90)
    
    for i, dim in enumerate(dims):
        vals = all_labels[:, i]
        mean_val = np.mean(vals)
        median_val = np.median(vals)
        std_val = np.std(vals)
        
        # 计算如果用 5.0 做阈值，正样本(>5)的比例
        pos_ratio_5 = np.sum(vals >= 5.0) / len(vals)
        
        # 推荐策略
        if 0.40 <= pos_ratio_5 <= 0.60:
            rec = "Use 5.0 (Balanced)"
        else:
            rec = f"Use Median ({median_val:.2f})"
            
        print(f"{dim:<12} | {mean_val:.2f}   | {median_val:.2f}   | {std_val:.2f}   | {pos_ratio_5:.2%}       | {rec}")
        
        stats_results[dim] = {
            'mean': mean_val,
            'median': median_val,
            'pos_ratio': pos_ratio_5
        }
        
    print("-" * 90)
    print("💡 分析解读: ")
    print("1. '>5.0 Ratio': 如果接近 50%，说明 5.0 是完美的阈值。")
    print("2. 如果 Ratio > 70% 或 < 30%，说明数据严重偏科。建议使用 Median 或 Focal Loss。")
    print("="*80 + "\n")
    return stats_results

# ---------------------------------------------------------
# 3. 辅助绘图函数
# ---------------------------------------------------------
def plot_head_outline(ax):
    """画头部轮廓"""
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, color='black', linewidth=2)
    ax.plot([0, 0.1, -0.1, 0], [1.0, 1.1, 1.1, 1.0], color='black', linewidth=2) # 鼻子
    ax.set_aspect('equal')
    ax.axis('off')

def visualize_graph_structure(data, title="Brain Connectivity Graph", ax=None):
    """1. 可视化脑网络"""
    if ax is None: fig, ax = plt.subplots()
    G = nx.Graph()
    for i, name in enumerate(CHANNEL_NAMES):
        if name in CHANNEL_POS: G.add_node(i, pos=CHANNEL_POS[name], label=name)
            
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy() if hasattr(data, 'edge_attr') else np.ones(edge_index.shape[1])
    
    # 去自环 + 仅显示最强 20% 连接
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    
    if len(edge_attr) > 0:
        threshold = np.percentile(edge_attr, 80)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            w = edge_attr[i]
            if w > threshold: G.add_edge(u, v, weight=w)

    pos = nx.get_node_attributes(G, 'pos')
    plot_head_outline(ax)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=0.5, edge_color=weights, edge_cmap=plt.cm.Blues)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='#ffcc99', edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_title(title, fontsize=14)

def visualize_connectivity_matrix(data, title="Adjacency Matrix", ax=None):
    """2. 可视化邻接矩阵"""
    if ax is None: fig, ax = plt.subplots()
    adj = np.zeros((32, 32))
    rows, cols = data.edge_index.numpy()
    weights = data.edge_attr.numpy() if hasattr(data, 'edge_attr') else np.ones(len(rows))
    adj[rows, cols] = weights
    sns.heatmap(adj, cmap="viridis", square=True, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title(title)

def visualize_de_features(data, ax=None):
    """3. 可视化 DE 特征"""
    if ax is None: fig, ax = plt.subplots()
    features = data.x.numpy()
    data_points = []
    for band_idx in range(5):
        vals = features[:, band_idx]
        for v in vals: data_points.append({'Band': BAND_NAMES[band_idx], 'DE Value': v})
    df = pd.DataFrame(data_points)
    sns.boxplot(x='Band', y='DE Value', data=df, ax=ax, palette="Set2")
    ax.set_title("DE Feature Distribution (Single Sample)")
    ax.grid(True, alpha=0.3)

def visualize_topomap(data, band_idx=2, title="Topomap", ax=None):
    """4. 脑地形图"""
    if ax is None: fig, ax = plt.subplots()
    values = data.x[:, band_idx].numpy()
    coords = np.array([CHANNEL_POS.get(name, (0, 0)) for name in CHANNEL_NAMES])
    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
    grid_z = griddata(coords, values, (grid_x, grid_y), method='cubic')
    plot_head_outline(ax)
    ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='jet', alpha=0.9, extend='both')
    ax.scatter(coords[:, 0], coords[:, 1], c='black', s=20, marker='.')
    ax.set_title(title)

def visualize_labels_distribution_advanced(data_list, stats, ax=None):
    """5. 高级标签分布图 (带中位数线)"""
    if ax is None: fig, ax = plt.subplots()
        
    all_labels = []
    # 随机抽样 2000 个样本
    sample_indices = np.random.choice(len(data_list), min(2000, len(data_list)), replace=False)
    for i in sample_indices:
        all_labels.append(data_list[i].y.numpy().flatten())
        
    all_labels = np.array(all_labels)
    df = pd.DataFrame(all_labels, columns=['Valence', 'Arousal', 'Dominance', 'Liking'])
    df_melt = df.melt(var_name='Dimension', value_name='Score')
    
    # 直方图
    sns.histplot(data=df_melt, x='Score', hue='Dimension', element="step", stat="density", common_norm=False, ax=ax, alpha=0.3)
    
    # 画标准线 (5.0)
    ax.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Threshold (5.0)')
    
    # 画各维度的中位数线
    colors = {'Valence': 'blue', 'Arousal': 'orange', 'Dominance': 'green', 'Liking': 'red'}
    y_min, y_max = ax.get_ylim()
    for dim, color in colors.items():
        median = stats[dim]['median']
        ax.axvline(x=median, ymin=0.9, ymax=1.0, color=color, linestyle='-', linewidth=3)
        ax.text(median, y_max*1.05, f'{dim[0]}', color=color, ha='center', fontsize=8, fontweight='bold')

    ax.set_title("Label Distribution with Median Indicators")
    
    # 自定义图例
    custom_lines = [Line2D([0], [0], color='red', linestyle='--'),
                    Line2D([0], [0], color='black', linestyle='-')]
    ax.legend(custom_lines, ['Standard (5.0)', 'Global Median (Top Marks)'], loc='upper left')

# ---------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------
def main():
    # 路径配置
    file_path_pearson = r"E:\Learning\大学课程\计算神经工程\LastWork\data\processed\deap_pearson_window1s.pt"
    file_path_plv = r"E:\Learning\大学课程\计算神经工程\LastWork\data\processed\deap_plv_window1s.pt"

    target_file = file_path_plv if os.path.exists(file_path_pearson) else file_path_plv
    
    print(f"Loading data from: {target_file}")
    try:
        data_list = torch.load(target_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # >>> 1. 先进行阈值分析 <<<
    stats = analyze_optimal_threshold(data_list)

    # >>> 2. 绘图 <<<
    sample = data_list[0]
    fig = plt.figure(figsize=(18, 12))
    
    # 子图 1-5
    ax1 = plt.subplot(2, 3, 1)
    visualize_graph_structure(sample, title="Brain Connectivity (Top 20%)", ax=ax1)
    
    ax2 = plt.subplot(2, 3, 2)
    visualize_connectivity_matrix(sample, title="Connectivity Matrix", ax=ax2)
    
    ax3 = plt.subplot(2, 3, 3)
    visualize_de_features(sample, ax=ax3)
    
    ax4 = plt.subplot(2, 3, 4)
    visualize_topomap(sample, band_idx=2, title="Alpha Band Topomap", ax=ax4)
    
    ax5 = plt.subplot(2, 3, 5)
    visualize_topomap(sample, band_idx=4, title="Gamma Band Topomap", ax=ax5)
    
    # 子图 6: 升级版标签分布
    ax6 = plt.subplot(2, 3, 6)
    visualize_labels_distribution_advanced(data_list, stats, ax=ax6)
    
    plt.suptitle(f"Data Visualization: {os.path.basename(target_file)}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(os.path.dirname(target_file), 'visualization_dashboard_v2.png')
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    sns.set_style("whitegrid")
    main()