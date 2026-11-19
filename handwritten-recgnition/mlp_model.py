
import numpy as np
import joblib
import time
from sklearn.neural_network import MLPClassifier

print("Loading dataset...")

data = np.load("processed_mnist.npz")
X_train = data["X_train"]
y_train = data["y_train"]

print("Dataset loaded:", X_train.shape)

print("\nCreating MLP model...")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    batch_size=128,
    learning_rate_init=0.001,
    max_iter=30,
    random_state=42,
    verbose=True,
    early_stopping=True,
    n_iter_no_change=5
)

print("\nTraining MLP model...")
start = time.time()

mlp.fit(X_train, y_train)

end = time.time()

print("\nMLP training completed!")
print("Training time:", round(end - start, 2), "seconds")
print("Epochs completed:", mlp.n_iter_)

joblib.dump(mlp, "mlp_model.pkl", compress=3)

print("MLP model saved as mlp_model.pkl")
