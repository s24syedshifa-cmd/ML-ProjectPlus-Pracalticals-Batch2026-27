
import numpy as np
import joblib
import time
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("=" * 55)
print("HANDWRITTEN DIGIT RECOGNITION - MODEL EVALUATION")
print("=" * 55)

# 1. Load test dataset
print("\nLoading test dataset...")

data = np.load("processed_mnist.npz")

X_test = data["X_test"]
y_test = data["y_test"]

print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)

# 2. Load saved models
models = {
    "KNN": "knn_model.pkl",
    "SVM": "svm_model.pkl",
    "MLP": "mlp_model.pkl"
}

results = []

# 3. Evaluate each model
for model_name, model_file in models.items():

    print("\n" + "=" * 55)
    print("Evaluating", model_name)
    print("=" * 55)

    print("Loading model...")
    model = joblib.load(model_file)

    print("Predicting test images...")
    start_time = time.time()

    y_pred = model.predict(X_test)

    prediction_time = time.time() - start_time

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1-score:", round(f1 * 100, 2), "%")
    print("Prediction time:", round(prediction_time, 2), "seconds")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save results for comparison
    results.append([
        model_name,
        round(accuracy * 100, 2),
        round(precision * 100, 2),
        round(recall * 100, 2),
        round(f1 * 100, 2),
        round(prediction_time, 2)
    ])

    # 4. Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=np.arange(10)
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(model_name + " Confusion Matrix")
    plt.tight_layout()

    image_file = model_name.lower() + "_confusion_matrix.png"
    plt.savefig(image_file)
    plt.close()

    print("Confusion matrix saved:", image_file)

# 5. Save comparison results to CSV
with open("model_comparison.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
        "Prediction Time (seconds)"
    ])

    writer.writerows(results)

# 6. Display final comparison
print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    f"{'Model':<10}"
    f"{'Accuracy':<13}"
    f"{'Precision':<13}"
    f"{'Recall':<13}"
    f"{'F1 Score':<13}"
)

for row in results:
    print(
        f"{row[0]:<10}"
        f"{row[1]:<13}"
        f"{row[2]:<13}"
        f"{row[3]:<13}"
        f"{row[4]:<13}"
    )

# 7. Identify best model by accuracy
best_model = max(results, key=lambda row: row[1])

print("\nBest model based on accuracy:", best_model[0])
print("Best accuracy:", best_model[1], "%")

print("\nComparison saved as model_comparison.csv")
print("\nPHASE 5 EVALUATION COMPLETED!")
