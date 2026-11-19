
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

print("Loading MNIST dataset...")

# Step 1: Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("\nOriginal Dataset Shapes")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Step 2: Normalize pixel values
# Convert pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("\nAfter Normalization")
print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())

# Step 3: Reshape images
# Convert each 28x28 image into 784 features
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

print("\nAfter Reshaping")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Step 4: Display one processed image
# Reshape one image back to 28x28 for visualization
plt.imshow(X_train[0].reshape(28, 28), cmap="gray")
plt.title("Processed MNIST Image - Label: " + str(y_train[0]))
plt.axis("off")
plt.show()

# Step 5: Save processed dataset
np.savez_compressed(
    "processed_mnist.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

print("\nProcessed dataset saved successfully!")
print("File created: processed_mnist.npz")
