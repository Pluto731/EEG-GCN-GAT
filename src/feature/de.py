# src/features/de.py

import numpy as np
from scipy.signal import butter, lfilter

# -----------------------------
# 1️⃣ 带通滤波函数
# -----------------------------
def bandpass_filter(signal, lowcut, highcut, fs=128, order=4):
    """
    Bandpass filter for EEG signal

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        1D EEG time series
    lowcut : float
        Low frequency of the band (Hz)
    highcut : float
        High frequency of the band (Hz)
    fs : int
        Sampling frequency
    order : int
        Order of the Butterworth filter

    Returns
    -------
    np.ndarray, shape (T,)
        Filtered signal
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = lfilter(b, a, signal)
    return filtered

# -----------------------------
# 2️⃣ Differential Entropy (DE)
# -----------------------------
def differential_entropy(signal):
    """
    Compute DE of a signal segment assuming Gaussian distribution

    Parameters
    ----------
    signal : np.ndarray, shape (T,)

    Returns
    -------
    float
        DE value
    """
    var = np.var(signal)
    if var <= 1e-10:
        var = 1e-10  # 避免 log(0)
    de = 0.5 * np.log(2 * np.pi * np.e * var)
    return de

# -----------------------------
# 3️⃣ 主函数：提取每个 segment 的 DE 特征
# -----------------------------
def extract_de_features(segment, fs=128):
    """
    Extract DE features for each EEG channel and standard frequency bands

    Parameters
    ----------
    segment : np.ndarray, shape (32, T)
        EEG segment
    fs : int
        Sampling frequency

    Returns
    -------
    np.ndarray, shape (32, 5)
        DE features per channel (δ, θ, α, β, γ)
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    n_channels = segment.shape[0]
    n_bands = len(bands)
    features = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        signal = segment[ch]
        for i, (band_name, (low, high)) in enumerate(bands.items()):
            filtered = bandpass_filter(signal, low, high, fs)
            features[ch, i] = differential_entropy(filtered)

    return features
