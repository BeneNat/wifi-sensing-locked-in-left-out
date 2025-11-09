import os, numpy as np, pandas as pd
from scipy.signal import medfilt, butter, filtfilt
from scipy.signal.windows import hann
from scipy.fft import rfft, rfftfreq
from collections import Counter
from parse_robust import parse_csi_from_file

# PARAMETERS
DATA_DIR = "data_utf8"              # Folder with csv
OUT_DIR = "data_features_improved"  # Folder for output features
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SEC = 10.0         # seconds per window
STEP_SEC = 5.0
TOP_SUBC = 16             # keep top N subcarriers by variance to reduce dim
BP_LOW = 0.05             # breathing low freq (Hz)
BP_HIGH = 0.6             # breathing high freq (Hz)
ASSUME_FS = None          # set to None to compute from timestamps

def bandpass(x, fs, low, high, order=3):
    nyq = 0.5*fs
    lown, highn = max(low/nyq, 1e-6), min(high/nyq, 0.999)
    b,a = butter(order, [lown, highn], btype='band')
    return filtfilt(b,a,x, axis=0)

def extract_window_features(window, fs):
    # window shape: (T, Nsub)
    feats = []
    # per-subcarrier stats (mean & std), then aggregate across subcarriers (mean, max)
    means = window.mean(axis=0)
    stds  = window.std(axis=0)
    feats.extend([means.mean(), means.std(), stds.mean(), stds.std(), stds.max()])
    # spatial variance (how much subcarriers differ)
    feats.append(np.var(means))
    # spectral features on averaged signal
    sig = window.mean(axis=1)
    n = len(sig)
    if n <= 2:
        feats.extend([0,0])
    else:
        w = sig * hann(n)
        spec = np.abs(rfft(w))
        freqs = rfftfreq(n, d=1.0/fs)
        # band-limited energy and dominant freq in band
        band_mask = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
        if band_mask.any():
            band_spec = spec[band_mask]
            band_freqs = freqs[band_mask]
            dom_idx = np.argmax(band_spec)
            feats.append(band_freqs[dom_idx])           # dominant freq in breathing band
            feats.append(band_spec.sum())               # band energy
        else:
            feats.extend([0,0])
    return np.array(feats, dtype=float)

def process_file(path, label_from_name):
    # parse using robust parser
    samples = parse_csi_from_file(path)
    if not samples:
        return [], []
    # convert to amplitude
    #data = np.array(samples, dtype=np.float32)   # shape (packets, subcarriers)
    # Filter packets so only those with consistent subcarrier length are kept
    samples = [s for s in samples if len(s) == 128]
    if len(samples) == 0:
        print(f"⚠️ No valid packets in {path}")
        return np.empty((0, 6)), np.empty((0,))  # skip empty

    data = np.array(samples, dtype=np.float32)

    # remove packets not matching most common length
    lengths = [len(s) for s in samples]
    most_common_len = Counter(lengths).most_common(1)[0][0]
    data = np.array([s[:most_common_len] for s in data if len(s)==most_common_len], dtype=np.float32)
    # amplitude: if data are already amplitudes, fine; if they are interleaved I/Q you need to reshape
    # compute per-subcarrier variance and pick top SUBC
    var = data.var(axis=0)
    top_idx = np.argsort(var)[-TOP_SUBC:]
    data_sel = data[:, top_idx]   # (T, TOP_SUBC)
    fs = 10.0  # fallback if timestamps unknown
    # sliding windows by packet count: choose window_len as int(WINDOW_SEC * fs)
    win_len = max(4, int(WINDOW_SEC * fs))
    step = max(1, int(STEP_SEC * fs))
    feats = []
    labels = []
    for start in range(0, max(1, len(data_sel)-win_len+1), step):
        w = data_sel[start:start+win_len]
        if w.shape[0] < win_len:
            continue
        # medfilt per-subcarrier to remove spikes
        w = medfilt(w, kernel_size=(3,1))
        fvec = extract_window_features(w, fs)
        feats.append(fvec)
        labels.append(label_from_name)
    return feats, labels

# MAIN: iterate files, create dataset
X_list, y_list, meta = [], [], []
for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith(".csv"):
        continue
    label = 1 if any(k in fname.lower() for k in ("person","breath","move")) else 0
    path = os.path.join(DATA_DIR, fname)
    feats, labels = process_file(path, label)
    if feats:
        X_list.extend(feats)
        y_list.extend(labels)
        meta.append((fname, len(feats)))
    print(f"Processed {fname} -> windows: {len(feats)}")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
print("Final dataset:", X.shape, y.shape)
np.save(os.path.join(OUT_DIR, "X_improved.npy"), X)
np.save(os.path.join(OUT_DIR, "y_improved.npy"), y)
pd.DataFrame(meta, columns=["file","windows"]).to_csv(os.path.join(OUT_DIR,"meta.csv"), index=False)
print("Saved improved features to", OUT_DIR)
