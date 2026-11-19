"""
Retrain the MLP with DATA AUGMENTATION (+ your own saved samples).

Run from the project folder (same folder as app.py):
    python train_mlp_augmented.py

Optional:
    python train_mlp_augmented.py --aug 3 --hidden 256,128 --epochs 40

It keeps the SAME pixel units as your processed_mnist.npz, so app.py needs
no change. The old model is backed up as mlp_model_backup.pkl.
"""
import os
import glob
import time
import shutil
import argparse

import numpy as np
import joblib
from PIL import Image
from scipy.ndimage import shift, rotate, zoom, grey_dilation, grey_erosion
from sklearn.neural_network import MLPClassifier

from utils.preprocessing import preprocess_image

parser = argparse.ArgumentParser()
parser.add_argument("--aug", type=int, default=2, help="augmented copies per MNIST image")
parser.add_argument("--hidden", type=str, default="256,128", help="hidden layer sizes")
parser.add_argument("--epochs", type=int, default=40, help="maximum epochs")
parser.add_argument("--own-repeat", type=int, default=30, help="copies of each of YOUR saved samples")
parser.add_argument("--limit", type=int, default=0, help="use only N training images (for quick tests)")
args = parser.parse_args()

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
data = np.load("processed_mnist.npz")
print("Keys in processed_mnist.npz:", list(data.keys()))
for key in ("X_train", "y_train", "X_test", "y_test"):
    if key not in data:
        raise SystemExit(f"'{key}' not found in processed_mnist.npz. Rename it in this script.")

X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
X_test = X_test.reshape(len(X_test), -1).astype(np.float32)
y_train = np.asarray(y_train).astype(int).ravel()
y_test = np.asarray(y_test).astype(int).ravel()

if args.limit:
    X_train, y_train = X_train[:args.limit], y_train[:args.limit]

lo, hi = float(X_train.min()), float(X_train.max())
print(f"Pixel range in your data: {lo} to {hi}")
if hi > 1.5:
    print("NOTE: your data is 0-255. It still works, but 0-1 normalised data usually trains better.")

def to01(x): return (x - lo) / (hi - lo)
def from01(x): return lo + x * (hi - lo)

# ---------------------------------------------------------------
# 2. AUGMENTATION  (all done on 28x28 images in the range 0-1)
# ---------------------------------------------------------------
def rescale(img, s):
    z = zoom(img, s, order=1)
    out = np.zeros((28, 28), dtype=np.float32)
    if s >= 1:
        start = (z.shape[0] - 28) // 2
        out[:, :] = z[start:start + 28, start:start + 28]
    else:
        start = (28 - z.shape[0]) // 2
        out[start:start + z.shape[0], start:start + z.shape[1]] = z
    return out

def augment(img):
    img = img.reshape(28, 28).astype(np.float32)
    img = rescale(img, rng.uniform(0.85, 1.15))
    img = rotate(img, rng.uniform(-12, 12), reshape=False, order=1)
    img = shift(img, (rng.uniform(-2, 2), rng.uniform(-2, 2)), order=1)
    r = rng.random()
    if r < 0.25:
        img = grey_dilation(img, size=(2, 2))      # thicker stroke
    elif r < 0.40:
        img = grey_erosion(img, size=(2, 2))       # thinner stroke
    return np.clip(img, 0, 1).reshape(784)

# ---------------------------------------------------------------
# 3. BUILD THE BIG TRAINING SET
# ---------------------------------------------------------------
X01 = to01(X_train)
parts_X, parts_y = [X01], [y_train]

print(f"Creating {args.aug} augmented copies of {len(X01)} images ...")
t0 = time.time()
for _ in range(args.aug):
    parts_X.append(np.array([augment(x) for x in X01], dtype=np.float32))
    parts_y.append(y_train)
print(f"  done in {time.time() - t0:.0f}s")

# Your own samples saved from the website (user_samples/<digit>/*.png)
own_X, own_y = [], []
for path in glob.glob(os.path.join("user_samples", "*", "*.png")):
    label = os.path.basename(os.path.dirname(path))
    if not label.isdigit():
        continue
    try:
        x = preprocess_image(Image.open(path), 0.0, 1.0).reshape(784)
    except Exception as e:
        print("  skipped", path, "->", e)
        continue
    for _ in range(args.own_repeat):
        own_X.append(augment(x))
        own_y.append(int(label))
if own_X:
    print(f"Added {len(own_X)} samples from YOUR handwriting "
          f"({len(own_X) // args.own_repeat} images x {args.own_repeat} copies)")
    parts_X.append(np.array(own_X, dtype=np.float32))
    parts_y.append(np.array(own_y))
else:
    print("No samples in user_samples/ yet (that is fine).")

X_all = from01(np.vstack(parts_X)).astype(np.float32)
y_all = np.concatenate(parts_y)
perm = rng.permutation(len(X_all))
X_all, y_all = X_all[perm], y_all[perm]
print("Total training images:", len(X_all))

# ---------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------
hidden = tuple(int(h) for h in args.hidden.split(","))
model = MLPClassifier(
    hidden_layer_sizes=hidden,
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=args.epochs,
    early_stopping=True,
    validation_fraction=0.05,
    n_iter_no_change=5,
    random_state=42,
    verbose=True,
)
print("Training ... this can take 10-30 minutes on a normal laptop.")
t0 = time.time()
model.fit(X_all, y_all)
print(f"Training finished in {(time.time() - t0) / 60:.1f} min")

acc = model.score(X_test, y_test) * 100
print(f"\nAccuracy on the standard MNIST test set: {acc:.2f}%")

# ---------------------------------------------------------------
# 5. SAVE  (old model is backed up)
# ---------------------------------------------------------------
if os.path.exists("mlp_model.pkl") and not os.path.exists("mlp_model_backup.pkl"):
    shutil.copy("mlp_model.pkl", "mlp_model_backup.pkl")
    print("Old model backed up as mlp_model_backup.pkl")
joblib.dump(model, "mlp_model.pkl")
print("New model saved as mlp_model.pkl. Restart app.py to use it.")