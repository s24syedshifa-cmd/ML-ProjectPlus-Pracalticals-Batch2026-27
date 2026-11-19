
import numpy as np

print("Loading processed MNIST dataset...")

# Load the saved dataset
data = np.load("processed_mnist.npz")

# Extract training and testing data
X_train = data["X_train"]
y_train = data["y_train"]

X_test = data["X_test"]
y_test = data["y_test"]

# Display dataset information
print("\nDataset Loaded Successfully!")

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nNumber of classes:", len(np.unique(y_train)))
print("Available digits:", np.unique(y_train))

print("\nReady for Machine Learning!")
