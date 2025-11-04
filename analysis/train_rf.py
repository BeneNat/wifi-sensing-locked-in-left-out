import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train = np.load("data_features/X_train.npy")
X_test = np.load("data_features/X_test.npy")
y_train = np.load("data_features/y_train.npy")
y_test = np.load("data_features/y_test.npy")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
model_path = "models/model_rf.joblib"
joblib.dump(clf, model_path)
print(f"\nModel saved to: {model_path}")
