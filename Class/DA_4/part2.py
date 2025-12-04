import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================
# 1. LOAD & LABEL DATA
# ============================================

# Load each motion class CSV file
df_waving = pd.read_csv("waving.csv", header=None)
df_vertical = pd.read_csv("vertical.csv", header=None)
df_stable = pd.read_csv("stable.csv", header=None)

colnames = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
df_waving.columns = colnames
df_vertical.columns = colnames
df_stable.columns = colnames


# Assign labels
df_waving["label"] = "waving"
df_vertical["label"] = "vertical"
df_stable["label"] = "stable"


# Merge datasets
df = pd.concat([df_waving, df_vertical, df_stable], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df.head())

# ============================================
# 2. FEATURE / LABEL EXTRACTION
# ============================================

# Extract IMU features
X = df[colnames].values
y = df['label'].values

# Encode labels → integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Normalize feature values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================================
# 3. DATA SPLITTING (60% train, 20% val, 20% test)
# ============================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ============================================
# 4. RESHAPE FOR 1DNN (samples, timesteps, features)
# ============================================

X_train = np.expand_dims(X_train, axis=1)
X_val   = np.expand_dims(X_val, axis=1)
X_test  = np.expand_dims(X_test, axis=1)

# ============================================
# 5. BUILD 1D NEURAL NETWORK
# ============================================

num_classes = len(np.unique(y))

model = keras.Sequential([
    layers.Input(shape=(1, 6)),       # 1 timestep, 6 features
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# 6. TRAIN MODEL
# ============================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32
)

# ============================================
# 7. ACCURACY & LOSS PLOTS
# ============================================

plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Val"])

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Val"])

plt.show()

# ============================================
# 8. TEST SET EVALUATION
# ============================================

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# ============================================
# 9. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=encoder.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ============================================
# 10. ROC CURVES (One-vs-Rest)
# ============================================

y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{encoder.classes_[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multi-Class)")
plt.legend()
plt.show()

# ============================================
# 11. SAVE MODEL (.h5 FORMAT)
# ============================================

model.save("motion_classifier.h5")
print("Model saved as motion_classifier.h5")