
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

from PIL import Image, UnidentifiedImageError
from utils.preprocessing import preprocess_image

app = Flask(__name__)

# Maximum uploaded image size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Load trained MLP model
MODEL_PATH = "mlp_model.pkl"
DATASET_PATH = "processed_mnist.npz"

print("Loading trained MLP model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("mlp_model.pkl not found!")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("processed_mnist.npz not found!")

model = joblib.load(MODEL_PATH)

# Load training pixel range
data = np.load(DATASET_PATH)

X_train = data["X_train"]
train_min = float(X_train.min())
train_max = float(X_train.max())

if model.n_features_in_ != 784:
    raise ValueError("Model must expect 784 input features.")

print("MLP model loaded successfully!")
print("Expected features:", model.n_features_in_)
print("Training pixel range:", train_min, "to", train_max)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Backend health check
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Backend is running!",
        "model": "MLPClassifier",
        "model_loaded": True,
        "input_features": model.n_features_in_
    })


# Digit prediction API
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Please select an image."
        }), 400

    try:
        image = Image.open(file.stream)

        processed_image = preprocess_image(
            image,
            train_min,
            train_max
        )

        # Predict digit
        prediction = model.predict(processed_image)[0]

        # Calculate confidence
        probabilities = model.predict_proba(processed_image)[0]
        confidence = float(np.max(probabilities)) * 100

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "confidence": round(confidence, 2),
            "message": "Digit predicted successfully!"
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "status": "error",
            "message": "Invalid image file."
        }), 400

    except Exception as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


# Handle oversized uploads
@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "status": "error",
        "message": "Image size must be less than 5 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
