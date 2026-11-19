
import numpy as np
import joblib
import time

from sklearn.svm import LinearSVC

print("Loading processed MNIST dataset...")

# Step 1: Load processed dataset
data = np.load("processed_mnist.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_test = data["X_test"]
y_test = data["y_test"]

print("Dataset loaded successfully!")

print("\nTraining dataset:", X_train.shape)
print("Testing dataset:", X_test.shape)

# Step 2: Create SVM model
print("\nCreating SVM model...")

svm = LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=42,
    dual="auto"
)

print("SVM model created successfully!")

# Step 3: Train SVM model
print("\nTraining SVM model...")
print("Please wait. Training may take some time.")

start_time = time.time()

svm.fit(X_train, y_train)

end_time = time.time()

print("\nSVM model training completed!")

print("Training time:",
      round(end_time - start_time, 2), "seconds")

# Step 4: Save trained SVM model
print("\nSaving SVM model...")

joblib.dump(svm, "svm_model.pkl", compress=3)

print("SVM model saved successfully!")
print("File created: svm_model.pkl")

print("\nSVM IMPLEMENTATION COMPLETED!")
