import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data_features/csi_features_clean.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature mean (train):", X_train_scaled.mean(axis=0))
print("Feature std (train):", X_train_scaled.std(axis=0))
print("First 5 labels:", y_train[:5])

np.save("data_features/X_train.npy", X_train_scaled)
np.save("data_features/X_test.npy", X_test_scaled)
np.save("data_features/y_train.npy", y_train)
np.save("data_features/y_test.npy", y_test)
print("[GOOD] Preprocessed data saved as .npy files")