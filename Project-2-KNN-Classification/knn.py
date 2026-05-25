# =========================================================
# DECODELABS AI PROJECT 2
# ADVANCED IRIS CLASSIFICATION SYSTEM
# Using K-Nearest Neighbors (KNN)
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =========================
# LOAD DATASET
# =========================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# Convert into DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("\n==============================")
print(" IRIS DATASET INFORMATION ")
print("==============================")

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nTarget Classes:")
for i, name in enumerate(target_names):
    print(f"{i} --> {name}")

# =========================
# CHECK FOR NULL VALUES
# =========================

print("\n==============================")
print(" CHECKING NULL VALUES ")
print("==============================")

print(df.isnull().sum())

# =========================
# FEATURE & TARGET SPLIT
# =========================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n==============================")
print(" TRAIN TEST SPLIT ")
print("==============================")

print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully!")

# =========================
# FIND BEST K VALUE
# =========================

error_rates = []

print("\n==============================")
print(" FINDING BEST K VALUE ")
print("==============================")

for k in range(1, 21):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train)

    y_pred_k = knn.predict(X_test_scaled)

    error = np.mean(y_pred_k != y_test)

    error_rates.append(error)

    print(f"K = {k} | Error Rate = {error:.4f}")

# =========================
# VISUALIZE ERROR RATE
# =========================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, 21),
    error_rates,
    marker='o',
    linestyle='dashed'
)

plt.title("Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")

plt.grid(True)

plt.show()

# =========================
# FINAL MODEL
# =========================

best_k = error_rates.index(min(error_rates)) + 1

print("\n==============================")
print(" BEST K VALUE ")
print("==============================")

print(f"Optimal K Value: {best_k}")

model = KNeighborsClassifier(n_neighbors=best_k)

# Train Model
model.fit(X_train_scaled, y_train)

print("\nModel Trained Successfully!")

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test_scaled)

# =========================
# EVALUATION METRICS
# =========================

print("\n==============================")
print(" MODEL EVALUATION ")
print("==============================")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy Score: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# Classification Report
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)

# =========================
# VISUALIZE CONFUSION MATRIX
# =========================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names
)

disp.plot(cmap='Blues')

plt.title("Confusion Matrix")

plt.show()

# =========================
# TEST CUSTOM FLOWER
# =========================

print("\n==============================")
print(" CUSTOM FLOWER PREDICTION ")
print("==============================")

# Example Flower:
# [sepal length, sepal width, petal length, petal width]

custom_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

# Scale Data
custom_flower_scaled = scaler.transform(custom_flower)

# Predict
prediction = model.predict(custom_flower_scaled)

predicted_class = target_names[prediction[0]]

print("\nFlower Features:")
print(custom_flower)

print(f"\nPredicted Flower Class: {predicted_class}")

# =========================
# MODEL SUMMARY
# =========================

print("\n==============================")
print(" PROJECT SUMMARY ")
print("==============================")

print(f"""
Algorithm Used       : K-Nearest Neighbors
Dataset              : Iris Dataset
Total Samples        : {len(df)}
Training Samples     : {len(X_train)}
Testing Samples      : {len(X_test)}
Optimal K Value      : {best_k}
Final Accuracy       : {accuracy:.4f}
""")

print("Project Execution Completed Successfully!")

# =========================================================
# END OF PROJECT
# =========================================================

