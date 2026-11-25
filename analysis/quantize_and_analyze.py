import os
import time
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and preprocess data (for quick test)
df = pd.read_csv("data_features/csi_features_clean.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 2. Load trained TinyML TorchScript model
print("Loading trained TinyML model...")
model = torch.jit.load("models/model_tinyml_mlp.pth")
model.eval()

# Evaluate before quantization
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()
y_pred_labels = (y_pred >= 0.5).astype(int)

acc_fp32 = accuracy_score(y_test, y_pred_labels)
print(f"\nOriginal FP32 Model Accuracy: {acc_fp32*100:.2f}%")

# 3. Quantize model (dynamic quantization)
print("\nPerforming dynamic quantization...")

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 4. Evaluate quantized model
quantized_model.eval()
with torch.no_grad():
    y_pred_q = quantized_model(X_test_tensor).numpy()
y_pred_labels_q = (y_pred_q >= 0.5).astype(int)

acc_q = accuracy_score(y_test, y_pred_labels_q)
print(f"Quantized Model Accuracy: {acc_q*100:.2f}%")
print("\nClassification Report (Quantized):\n", classification_report(y_test, y_pred_labels_q))

# 5. Save quantized model (TorchScript)
os.makedirs("models", exist_ok=True)
example_input = torch.rand(1, X_test.shape[1])
traced_quant_model = torch.jit.trace(quantized_model, example_input)
traced_quant_model.save("models/model_tinyml_mlp_quantized.pth")
print("\nQuantized model saved to models/model_tinyml_mlp_quantized.pth")

# 6. Analyze model size and parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

fp32_size = os.path.getsize("models/model_tinyml_mlp.pth") / 1024
int8_size = os.path.getsize("models/model_tinyml_mlp_quantized.pth") / 1024

print("\nModel Comparison:")
print(f" - FP32 Model Size: {fp32_size:.2f} KB")
print(f" - Quantized Model Size: {int8_size:.2f} KB")
print(f" - Compression Ratio: {int8_size/fp32_size:.2f}x smaller")

try:
    n_params = count_parameters(model)
    print(f" - Trainable Parameters: {n_params:,}")
except:
    print("[WARNING] Parameter count unavailable (TorchScript format)")

param_bytes = n_params * 4  # FP32 → 4 bytes per parameter
param_bytes_q = n_params  # INT8 → 1 byte per parameter
print(f" - Estimated RAM (FP32): {param_bytes/1024:.2f} KB")
print(f" - Estimated RAM (INT8): {param_bytes_q/1024:.2f} KB")

N = 1000
start = time.time()
for _ in range(N):
    _ = model(X_test_tensor[:1])
fp32_time = (time.time() - start) / N

start = time.time()
for _ in range(N):
    _ = quantized_model(X_test_tensor[:1])
q_time = (time.time() - start) / N

print(f"\nAvg FP32 inference time: {fp32_time*1000:.3f} ms")
print(f"Avg Quantized inference time: {q_time*1000:.3f} ms")

print("\nQuantization and analysis completed successfully.")
