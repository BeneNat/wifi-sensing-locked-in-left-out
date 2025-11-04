# server/live_features.py
import os
import json
import numpy as np
import torch
from joblib import load as joblib_load
from analysis.features import extract_features_from_window

class LiveFeatureBuffer:
    """ Maintains a rolling window of CSI payloads and outputs feature vectors. """

    def __init__(self, window_size=60, step=25, n_subcarriers=None):
        self.window_size = window_size
        self.step = step
        self.buffer = []
        self.n_sub = n_subcarriers
        self.model, self.model_info = self._load_active_model()

    # ----------------------------------------------------------------------
    def _load_active_model(self):
        meta_path = os.path.join("models", "model_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Missing models/model_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        active_name = meta.get("active_model")
        model_info = meta["models"].get(active_name)
        if model_info is None:
            raise ValueError(f"Active model '{active_name}' not found.")

        model_path = os.path.join("models", active_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"🧩 Loading model: {active_name} ({model_info['type']})")

        model_type = model_info["type"]
        if model_type == "rf":
            model = joblib_load(model_path)
            print("✅ Loaded RandomForest model")
        elif "mlp" in model_type:
            try:
                model = torch.jit.load(model_path, map_location="cpu")
                print("✅ Loaded TorchScript / quantized model (torch.jit.load)")
            except Exception as e:
                print("⚠️ torch.jit.load failed:", e)
                model = torch.load(model_path, map_location="cpu", weights_only=False)
                print("✅ Loaded fallback PyTorch model (torch.load)")
            model.eval()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return model, model_info

    # ----------------------------------------------------------------------
    def add_sample(self, payload):
        if not isinstance(payload, np.ndarray):
            payload = np.array(payload, dtype=np.float32)

        if self.n_sub is None:
            self.n_sub = int(payload.shape[0])
        elif payload.shape[0] != self.n_sub:
            if payload.shape[0] > self.n_sub:
                payload = payload[:self.n_sub]
            else:
                payload = np.pad(payload, (0, self.n_sub - payload.shape[0]))

        self.buffer.append(payload)

        if len(self.buffer) >= self.window_size and len(self.buffer) % self.step == 0:
            window = np.stack(self.buffer[-self.window_size:], axis=0)
            feats = extract_features_from_window(window)
            if np.std(window) < 1e-3:
                print("[WARN] CSI window nearly constant! Check STA/AP data stream.")

            # Normalize before inference (TEMP SCALING TEST)
            feats = np.nan_to_num(feats)

            # 🔧 quick runtime scaling to match training scale
            feats = feats / 1e6  # try 1e5 first; if still huge, try 1e6 later

            # Skip local normalization for now
            # feats_mean = np.mean(feats)
            # feats_std = np.std(feats) + 1e-6
            # feats = (feats - feats_mean) / feats_std

            print(f"[DEBUG_FEATS] mean={np.mean(feats):.6f} std={np.std(feats):.6f} first5={feats[:5]}")

            pred = self._predict(feats)
            return feats, pred
        return None, None

    # ----------------------------------------------------------------------
    def _predict(self, features):
        model_type = self.model_info["type"]
        try:
            if model_type == "rf":
                pred_val = int(self.model.predict([features])[0])
                prob = getattr(self.model, "predict_proba", lambda x: [[None]])([features])[0]
                conf = float(max(prob)) if prob and prob[0] is not None else None
                return {"pred": pred_val, "conf": conf}

            elif "mlp" in model_type:
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = self.model(x)
                print(f"[DEBUG_MODEL_RAW_OUT] {out}")
                print(f"[DEBUG_MODEL_OUT] type={type(out)} shape={getattr(out, 'shape', 'N/A')} repr={str(out)[:120]}")

                if isinstance(out, dict):
                    out = next(iter(out.values()))
                elif isinstance(out, (list, tuple)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    out = torch.tensor(out, dtype=torch.float32)
                out = out.cpu()
                if out.dim() == 1:
                    out = out.unsqueeze(0)

                if out.shape[1] == 1:
                    prob1 = torch.sigmoid(out).item()
                    #val = out.item()
                    ## if it already looks like a probability, skip sigmoid
                    #if 0.0 <= val <= 1.0:
                    #    prob1 = val
                    #else:
                    #    prob1 = torch.sigmoid(torch.tensor(val)).item()
                    pred = 1 if prob1 >= 0.7 else 0
                    conf = float(prob1 if pred == 1 else 1.0 - prob1)
                    print(f"[DEBUG_PROB] prob1={prob1:.4f}")
                    return {"pred": int(pred), "conf": float(conf)}
                else:
                    probs = torch.softmax(out, dim=1).numpy()[0]
                    pred = int(np.argmax(probs))
                    conf = float(np.max(probs))
                    print(f"[DEBUG_PROBS] probs[:5]={probs[:5]} pred={pred} conf={conf:.4f}")
                    return {"pred": pred, "conf": conf}

            else:
                return {"pred": None, "conf": None}
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return {"pred": None, "conf": None}
