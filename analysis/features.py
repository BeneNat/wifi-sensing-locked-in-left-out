import numpy as np
from scipy.signal import medfilt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

def extract_features_from_window(window):
    mean_vals = np.mean(window, axis=0)
    std_vals = np.std(window, axis=0)
    energy = np.mean(window**2, axis=0)
    entropy = -np.sum((window**2) * np.log(window**2 + 1e-9), axis=0)

    with np.errstate(all='ignore'):
        skew_vals = skew(window, axis=0, nan_policy='omit')
        kurt_vals = kurtosis(window, axis=0, nan_policy='omit')

    fft_vals = np.abs(np.fft.rfft(window, axis=0))
    dom_freq = np.nanmean(np.argmax(fft_vals, axis=0))
    spectral_energy = np.nanmean(np.sum(fft_vals**2, axis=0))

    diffs = np.diff(window, axis=0)
    mean_diff = np.nanmean(np.abs(diffs))
    var_diff = np.nanvar(diffs)

    feats_global = [
        np.nanmean(mean_vals), np.nanstd(mean_vals),
        np.nanmean(std_vals), np.nanstd(std_vals),
        np.nanmean(skew_vals), np.nanmean(kurt_vals),
        np.nanmean(energy), np.nanmean(entropy),
        dom_freq, spectral_energy,
        mean_diff, var_diff
    ]

    n_sub = window.shape[1]
    band_edges = np.linspace(0, n_sub, 5, dtype=int)
    feats_bands = []
    for i in range(4):
        b = window[:, band_edges[i]:band_edges[i+1]]
        diffs_b = np.diff(b, axis=0)
        feats_bands.extend([
            np.nanmean(b), np.nanstd(b),
            np.nanmean(diffs_b), np.nanvar(diffs_b)
        ])
    
    feats = np.array(feats_global + feats_bands, dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def preprocess_csi(samples, window_size=50, step=25):
    """Offline batch preprocessing: from list of packets (each 1D array) -> feature vectors"""
    from collections import Counter
    lengths = [len(s) for s in samples]
    if len(set(lengths)) > 1:
        most_common_len = Counter(lengths).most_common(1)[0][0]
        samples = [s for s in samples if len(s) == most_common_len]

    samples = np.array(samples)
    if samples.ndim != 2:
        raise ValueError("CSI samples must be 2D (packets x subcarriers)")

    samples = medfilt(samples, kernel_size=(3, 1))
    samples = np.clip(samples, -5, 5)

    scaler = StandardScaler()
    samples_norm = scaler.fit_transform(samples)

    features = []
    for start in range(0, len(samples_norm) - window_size, step):
        window = samples_norm[start:start + window_size, :]
        feats = extract_features_from_window(window)
        features.append(feats)

    return np.array(features)
