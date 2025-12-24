# src/features/de.py

import numpy as np

# -----------------------------
# Differential Entropy (DE)
# -----------------------------
def differential_entropy(signal):
    """
    Compute Differential Entropy (DE) assuming Gaussian distribution.

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
        var = 1e-10
    return 0.5 * np.log(2 * np.pi * np.e * var)


# -----------------------------
# Extract DE features (DEAP preprocessed EEG)
# -----------------------------
def extract_de_features(segment):
    """
    Extract DE features from DEAP preprocessed EEG segments.

    Notes
    -----
    The DEAP dataset has already been band-pass filtered (0.5–45 Hz).
    Therefore, no additional band-pass filtering is applied here.
    The same DE value is replicated to form a 5-dimensional feature
    for compatibility with frequency-band-based feature formats.

    Parameters
    ----------
    segment : np.ndarray, shape (32, T)

    Returns
    -------
    np.ndarray, shape (32, 5)
        DE features per channel
    """
    n_channels = segment.shape[0]
    n_bands = 5  # placeholder for δ, θ, α, β, γ

    features = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        de = differential_entropy(segment[ch])
        features[ch, :] = de

    return features
