// ======================================
// HANDWRITTEN DIGIT RECOGNITION - FRONTEND
// Sends images to the Flask backend as multipart FormData
// (field name "image", exactly what app.py reads)
// ======================================

// If you open the site from Flask (http://127.0.0.1:5000) leave this EMPTY.
// Only if you open the page from another server (Live Server, etc.) set it to
// "http://127.0.0.1:5000" (and install flask-cors in the backend).
const API_BASE = "";
const SCRIPT_VERSION = "v2";

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");

const imageUpload = document.getElementById("imageUpload");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const uploadPredictBtn = document.getElementById("uploadPredictBtn");
const clearUploadBtn = document.getElementById("clearUploadBtn");

const prediction = document.getElementById("prediction");
const resultTitle = document.getElementById("resultTitle");
const resultMessage = document.getElementById("resultMessage");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");
const statusBox = document.getElementById("status"); // renamed: "status" clashes with window.status

let isDrawing = false;
let hasDrawing = false;
let selectedFile = null;   // the uploaded image file
let isBusy = false;

// ======================================
// BUTTON STATES
// ======================================

function updateButtons() {
    predictBtn.disabled = isBusy;
    uploadPredictBtn.disabled = isBusy || !selectedFile;
    clearUploadBtn.disabled = isBusy;
    clearBtn.disabled = isBusy;
}

// ======================================
// CANVAS SETTINGS
// ======================================

function initializeCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "white";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
}

initializeCanvas();

function getPosition(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height
    };
}

canvas.addEventListener("pointerdown", function (event) {
    isDrawing = true;
    hasDrawing = true;
    canvas.setPointerCapture(event.pointerId);

    const pos = getPosition(event);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.lineTo(pos.x + 0.1, pos.y + 0.1);
    ctx.stroke();
});

canvas.addEventListener("pointermove", function (event) {
    if (!isDrawing) return;
    const pos = getPosition(event);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
});

function stopDrawing() {
    isDrawing = false;
    ctx.closePath();
}

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("lostpointercapture", stopDrawing);

// ======================================
// RESULT HELPERS
// ======================================

function resetResult() {
    prediction.textContent = "?";
    resultTitle.textContent = "Waiting for Input";
    resultMessage.textContent = "Draw or upload a digit to begin recognition.";
    confidence.textContent = "--";
    confidenceBar.style.width = "0%";
}

function displayResult(data) {
    prediction.textContent = data.prediction;
    resultTitle.textContent = "Prediction Successful";
    resultMessage.textContent = "The MLP model recognized your handwritten digit.";
    confidence.textContent = data.confidence + "%";
    confidenceBar.style.width = Math.min(data.confidence, 100) + "%";
    statusBox.textContent = "● Prediction completed successfully";
}

// ======================================
// CLEAR DRAWING CANVAS
// ======================================

clearBtn.addEventListener("click", function () {
    initializeCanvas();
    hasDrawing = false;
    resetResult();
    statusBox.textContent = "● Canvas cleared. Draw the next digit.";
});

// ======================================
// CLEAR UPLOADED IMAGE  (NEW)
// ======================================

function clearUpload() {
    selectedFile = null;
    imageUpload.value = "";               // lets you pick the same file again
    imagePreview.removeAttribute("src");
    previewContainer.style.display = "none";
    resetResult();
    statusBox.textContent = "● Image cleared. Upload the next image.";
    updateButtons();
}

clearUploadBtn.addEventListener("click", clearUpload);

// ======================================
// SEND IMAGE TO FLASK  (FIXED)
//  - multipart FormData with field name "image"
//  - reads data.status / data.message like app.py returns
// ======================================

async function predictDigit(imageBlob, fileName) {
    isBusy = true;
    updateButtons();

    statusBox.textContent = "● Sending image to MLP model...";
    resultTitle.textContent = "Processing...";
    resultMessage.textContent =
        "Please wait while the machine learning model predicts the digit.";

    try {
        const formData = new FormData();
        formData.append("image", imageBlob, fileName);

        const response = await fetch(API_BASE + "/predict", {
            method: "POST",
            body: formData            // do NOT set Content-Type manually
        });

        let data = {};
        try { data = await response.json(); } catch (e) { /* non-JSON reply */ }

        if (!response.ok || data.status !== "success") {
            throw new Error(data.message || "Prediction failed (server error).");
        }

        displayResult(data);

    } catch (error) {
        console.error("Prediction error:", error);
        prediction.textContent = "?";
        resultTitle.textContent = "Prediction Failed";
        resultMessage.textContent =
            error instanceof TypeError
                ? "Cannot reach the backend. Is the Flask server running?"
                : error.message;
        confidence.textContent = "--";
        confidenceBar.style.width = "0%";
        statusBox.textContent = "● Backend prediction error";
    } finally {
        isBusy = false;
        updateButtons();
    }
}

// ======================================
// PREDICT DRAWN DIGIT
// ======================================

predictBtn.addEventListener("click", function () {
    if (!hasDrawing) {
        alert("Please draw a digit first.");
        return;
    }
    canvas.toBlob(function (blob) {
        if (!blob) {
            alert("Could not read the drawing. Please try again.");
            return;
        }
        predictDigit(blob, "drawing.png");
    }, "image/png");
});

// ======================================
// UPLOAD IMAGE
// ======================================

imageUpload.addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (!file) return;

    const allowedTypes = ["image/png", "image/jpeg"];
    if (!allowedTypes.includes(file.type)) {
        alert("Please upload a PNG or JPG image.");
        imageUpload.value = "";
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        alert("Image size must be less than 5 MB.");
        imageUpload.value = "";
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = function (e) {
        imagePreview.src = e.target.result;
        previewContainer.style.display = "block";
        resetResult();
        statusBox.textContent = "● Image uploaded. Click Predict Uploaded Image.";
        updateButtons();
    };
    reader.readAsDataURL(file);

    // allow choosing the same file again later
    imageUpload.value = "";
});

// ======================================
// PREDICT UPLOADED IMAGE
// ======================================

uploadPredictBtn.addEventListener("click", function () {
    if (!selectedFile) {
        alert("Please upload an image first.");
        return;
    }
    predictDigit(selectedFile, selectedFile.name);
});

// ======================================
// TEST BACKEND CONNECTION
// ======================================

async function checkBackend() {
    const url = API_BASE + "/api/health";
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error("HTTP " + response.status + " from " + url);
        }
        const data = await response.json();
        if (data.status !== "success") {
            throw new Error("Unexpected reply from " + url);
        }
        statusBox.textContent = "● Backend connected successfully (script " + SCRIPT_VERSION + ")";
    } catch (error) {
        statusBox.textContent =
            "● Backend not connected: " + error.message +
            ". Run app.py and open http://127.0.0.1:5000";
        console.error("Backend error:", error);
    }
}

statusBox.textContent = "● Script " + SCRIPT_VERSION + " loaded. Checking backend...";
updateButtons();
checkBackend();
console.log("Digit Recognition Frontend Loaded!");