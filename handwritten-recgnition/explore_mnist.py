
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

print("Loading MNIST dataset...")
print("Please wait...")

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("\nMNIST Dataset Loaded Successfully!")

# Display dataset information
print("\nDataset Information")
print("--------------------")

print("Training Images:", X_train.shape[0])
print("Testing Images:", X_test.shape[0])
print("Total Images:", len(X_train) + len(X_test))
print("Image Dimensions:", X_train.shape[1], "x", X_train.shape[2])
print("Training Labels:", len(y_train))
print("Testing Labels:", len(y_test))

# Display first 10 handwritten digits
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title("Label: " + str(y_train[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()

# Display numerical pixel values
print("\nFirst Image Pixel Values:")
print(X_train[0])

print("\nFirst Image Label:", y_train[0])

print("\nArray Shapes")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nFirst 10 Training Labels:")
print(y_train[:10])

print("\nFirst Training Image:")
print(X_train[0])