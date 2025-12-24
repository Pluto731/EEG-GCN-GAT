# src/features/connectivity.py

import numpy as np
from scipy.signal import hilbert

# -----------------------------
# 1️⃣ Pearson 相关系数
# -----------------------------
def pearson_connectivity(segment):
    """
    Compute Pearson correlation matrix for EEG segment

    Parameters
    ----------
    segment : np.ndarray, shape (channels, T)

    Returns
    -------
    np.ndarray, shape (channels, channels)
        Pearson correlation adjacency matrix
    """
    corr = np.corrcoef(segment)
    # 将 NaN 替换为 0，防止后续图神经网络报错
    return np.nan_to_num(corr)


# -----------------------------
# 2️⃣ PLV (Phase Locking Value)
# -----------------------------

def plv_connectivity(segment):
    """
    Compute PLV adjacency matrix (Vectorized version)
    
    Parameters
    ----------
    segment : np.ndarray, shape (channels, T)
    """
    n_channels = segment.shape[0]
    
    # 1. 获取解析信号并提取相位
    # axis=1 表示沿时间轴做 Hilbert 变换
    analytic_signal = hilbert(segment, axis=1)
    phase = np.angle(analytic_signal)  # shape: (channels, T)
    
    # 2. 计算复数单位向量: e^(i * phase)
    # shape: (channels, T)
    complex_phase = np.exp(1j * phase)
    
    # 3. 利用矩阵乘法一次性计算所有通道对的相位差平均
    # 矩阵乘法: (channels, T) @ (T, channels) -> (channels, channels)
    # 共轭转置 (conj().T) 相当于在指数中做了减法: e^(i*a) * e^(-i*b) = e^(i*(a-b))
    # 最后除以时间点数 T 进行平均
    T = segment.shape[1]
    plv_matrix = np.abs(np.dot(complex_phase, complex_phase.conj().T) / T)
    
    return plv_matrix


# -----------------------------
# 3️⃣ 可选 threshold
# -----------------------------
def threshold_matrix(adj, thresh=0.5):
    """
    Apply threshold to adjacency matrix
    - Values below threshold set to 0

    Parameters
    ----------
    adj : np.ndarray, shape (channels, channels)
    thresh : float
        Threshold value between 0 and 1

    Returns
    -------
    np.ndarray, shape (channels, channels)
        Thresholded adjacency matrix
    """
    adj_thresh = adj.copy()
    adj_thresh[adj_thresh < thresh] = 0
    return adj_thresh
