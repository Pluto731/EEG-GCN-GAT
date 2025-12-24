# src/features/de.py

import numpy as np
from scipy.signal import welch

# -----------------------------
# Differential Entropy (DE) Computation
# -----------------------------

def calculate_de_from_variance(variance):
    """
    Apply the Differential Entropy formula for Gaussian distribution.
    Formula: H(X) = 0.5 * log(2 * pi * e * sigma^2)
    """
    # 数值稳定性保护：防止 variance 为 0 或负数导致 log 报错
    variance = np.maximum(variance, 1e-10)
    return 0.5 * np.log(2 * np.pi * np.e * variance)

def differential_entropy_features(signal, fs=128):
    """
    Compute Differential Entropy (DE) features in different frequency bands.
    Based on Power Spectral Density (PSD).

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
    fs : int, sampling frequency

    Returns
    -------
    np.ndarray, shape (5,)
        DE features for [delta, theta, alpha, beta, gamma]
    """
    # Frequency bands definition
    freq_bands = {
        'delta': [0.5, 4],
        'theta': [4, 8],
        'alpha': [8, 14],
        'beta': [14, 31],
        'gamma': [31, 45]
    }

    # 1. Compute PSD using Welch method
    # nperseg 可以根据信号长度调整，通常取 fs 或 2*fs
    freqs, psd = welch(signal, fs=fs, nperseg=fs)

    features = np.zeros(5)
    
    for i, (band_name, (f_low, f_high)) in enumerate(freq_bands.items()):
        # Find indices of frequencies in the band
        idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
        
        # 2. Calculate Band Power (which approximates Variance sigma^2)
        if np.sum(idx_band) > 0:
            # 使用积分计算频带内的总功率
            power_in_band = np.trapz(psd[idx_band], freqs[idx_band])
            
            # 3. Apply DE formula: DE = 0.5 * log(2*pi*e * power)
            # 在频域中，Band Power 等价于时域滤波后的 Variance
            features[i] = calculate_de_from_variance(power_in_band)
        else:
            # 如果该频带没有能量（罕见），给一个极小值的 DE
            features[i] = calculate_de_from_variance(1e-10)

    return features

# -----------------------------
# Extract Features from Segment (Multi-channel)
# -----------------------------
def extract_de_features(segment, fs=128):
    """
    Extract DE features from EEG segments for all channels.

    Parameters
    ----------
    segment : np.ndarray, shape (n_channels, n_samples)
        e.g., (32, 8064) for DEAP

    Returns
    -------
    np.ndarray, shape (n_channels, 5)
        DE features per channel per band.
    """
    n_channels = segment.shape[0]
    n_bands = 5  # δ, θ, α, β, γ

    features = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        # 对每个通道计算 5 个频带的 DE 值
        features[ch, :] = differential_entropy_features(segment[ch], fs=fs)
        
        # [注] 移除了内部的线性归一化 (x / max)。
        # 原因：DE 特征是经过 log 变换的，值域通常在 -2 到 5 之间。
        # 单个样本内的简单除法归一化会消除能量幅值的物理意义。
        # 建议在数据集构建完成后，对整个数据集进行 StandardScaler (Z-score)。

    return features