# src/features/connectivity.py

import numpy as np

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
    return np.corrcoef(segment)  # shape: (channels, channels)


# -----------------------------
# 2️⃣ PLV (Phase Locking Value)
# -----------------------------
from scipy.signal import hilbert

def plv_connectivity(segment):
    """
    Compute PLV adjacency matrix for EEG segment

    Parameters
    ----------
    segment : np.ndarray, shape (channels, T)

    Returns
    -------
    np.ndarray, shape (channels, channels)
        PLV adjacency matrix
    """
    n_channels = segment.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))
    
    # 计算每个通道的瞬时相位
    phase = np.angle(hilbert(segment, axis=1))  # shape: (channels, T)
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            phase_diff = phase[i] - phase[j]
            plv = np.abs(np.sum(np.exp(1j * phase_diff)) / len(phase_diff))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv  # 对称矩阵
    
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
