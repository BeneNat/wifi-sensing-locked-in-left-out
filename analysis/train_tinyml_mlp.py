# fourth_step_train_tinyml_mlp.py
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------
# 1. Load preprocessed CSV
# -------------------------------
df = pd.read_csv("data_features/csi_features_clean.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# -------------------------------
# 2. Split train/test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 3. Standardize features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. Convert to PyTorch tensors
# -------------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# -------------------------------
# 5. Create DataLoader
# -------------------------------
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -------------------------------
# 6. Define TinyML-friendly MLP
# -------------------------------
class TinyML_MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),   # small hidden layer
            nn.ReLU(),
            nn.Linear(32, 16),          # even smaller
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# -------------------------------
# 7. Initialize model, loss, optimizer
# -------------------------------
model = TinyML_MLP(input_dim=X_train_scaled.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 8. Training loop
# -------------------------------
epochs = 50
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test_tensor)
        val_loss = criterion(val_preds, y_test_tensor).item()

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# -------------------------------
# 9. Evaluation
# -------------------------------
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_labels = (y_pred >= 0.5).int().numpy()

acc = accuracy_score(y_test, y_pred_labels)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))

# -------------------------------
# 10. Confusion Matrix + ROC
# -------------------------------
cm = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No person", "Person"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – TinyML MLP")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve – TinyML MLP")
plt.show()

# -------------------------------
# Save TinyML TorchScript model
# -------------------------------
os.makedirs("models", exist_ok=True)
# example input for tracing (1 sample, num_features)
example_input = torch.rand(1, X_train_scaled.shape[1])
# trace the model
traced_model = torch.jit.trace(model, example_input)
# save traced model
traced_model.save("models/tinyml_mlp_model.pth")
print("✅ TinyML TorchScript model saved to models/tinyml_mlp_model.pth")

