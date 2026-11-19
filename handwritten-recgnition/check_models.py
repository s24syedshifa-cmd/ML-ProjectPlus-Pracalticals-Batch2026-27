
import os

models = [
    "knn_model.pkl",
    "svm_model.pkl",
    "mlp_model.pkl"
]

print("Checking saved models...\n")

for model in models:
    if os.path.exists(model):
        print(model, "- Saved successfully")
    else:
        print(model, "- Not found")

print("\nPhase 4 verification completed!")
