
import numpy as np
import joblib
import time

from sklearn.neighbors import KNeighborsClassifier

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

# Step 2: Create KNN model
print("\nCreating KNN model...")

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean",
    weights="uniform",
    n_jobs=-1
)

print("KNN model created successfully!")

# Step 3: Train KNN model
print("\nTraining KNN model...")

start_time = time.time()

knn.fit(X_train, y_train)

end_time = time.time()

print("KNN model training completed!")

print("Training time:", round(end_time - start_time, 2), "seconds")

# Step 4: Save trained model
print("\nSaving KNN model...")

joblib.dump(knn, "knn_model.pkl", compress=3)

print("KNN model saved successfully!")
print("File created: knn_model.pkl")

print("\nKNN IMPLEMENTATION COMPLETED!")
