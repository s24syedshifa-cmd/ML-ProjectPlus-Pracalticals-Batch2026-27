import numpy as np
from PIL import Image


def preprocess_image(image, train_min=0.0, train_max=1.0):
    """
    Convert any uploaded or drawn image into the 28x28 MNIST-style input
    (shape (1, 784)) that the MLP expects.

    Handles: transparent PNGs, colour photos, dark-on-light paper digits,
    light-on-dark canvas digits, off-centre digits and different sizes.
    """
    # 1. Flatten transparency onto a white background, then go grayscale
    image = image.convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(white, image).convert("L")

    arr = np.asarray(image, dtype=np.float32) / 255.0

    # 2. MNIST digits are WHITE on BLACK. If the border is bright
    #    (paper-style image), invert it.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() > 0.5:
        arr = 1.0 - arr

    # 3. Stretch contrast and remove faint background noise
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-3:
        raise ValueError("The image looks blank. Please use a clear digit.")
    arr = (arr - lo) / (hi - lo)
    arr[arr < 0.2] = 0.0

    # 4. Crop tightly around the digit
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        raise ValueError("No digit found in the image.")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 5. Resize so the longest side is 20 px (keeps aspect ratio, like MNIST)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    small = np.asarray(small, dtype=np.float32) / 255.0

    # 6. Place it on a 28x28 canvas, centred by centre of mass (like MNIST)
    total = small.sum()
    yy, xx = np.indices(small.shape)
    cy = (yy * small).sum() / total
    cx = (xx * small).sum() / total
    x0 = int(round(14 - cx))
    y0 = int(round(14 - cy))
    x0 = min(max(x0, 0), 28 - new_w)
    y0 = min(max(y0, 0), 28 - new_h)

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[y0:y0 + new_h, x0:x0 + new_w] = small

    # 7. Scale to the same pixel range the model was trained on
    canvas = train_min + canvas * (train_max - train_min)

    return canvas.reshape(1, 784)