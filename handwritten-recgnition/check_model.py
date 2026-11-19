
import numpy as np
import joblib

print("====================================")
print(" HANDWRITTEN DIGIT MODEL INSPECTION")
print("====================================")

# Check dataset
print("\n1. Checking processed dataset...")

data = np.load("processed_mnist.npz")

X_train = data["X_train"]
X_test = data["X_test"]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())
print("Data type:", X_train.dtype)

# Load MLP model
print("\n2. Loading trained MLP model...")

model = joblib.load("mlp_model.pkl")

print("Model loaded successfully!")

# Model information
print("\n3. MLP Model Details")

print("Model type:", type(model).__name__)
print("Expected features:", model.n_features_in_)
print("Hidden layers:", model.hidden_layer_sizes)
print("Activation:", model.activation)
print("Solver:", model.solver)
print("Number of classes:", len(model.classes_))

print("\n4. Testing model prediction...")

sample = X_test[:1]

prediction = model.predict(sample)

print("Actual digit:", data["y_test"][0])
print("Predicted digit:", prediction[0])

print("\n====================================")
print(" MODEL INSPECTION COMPLETED")
print("====================================")
