from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/custom_cnn_best.keras")
ROTTEN_DIR = Path(r"D:\FreshFruit Ai\dataset\test\Rotten")

IMG_SIZE = (160, 160)

model = tf.keras.models.load_model(MODEL_PATH)

images = []
paths = []

for path in sorted(ROTTEN_DIR.iterdir()):
    if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = Image.open(path).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image, dtype=np.float32)

    images.append(image)
    paths.append(path)

X = np.array(images)

predictions = model.predict(X, verbose=0).reshape(-1)

fresh_count = 0
rotten_count = 0

for path, prob in zip(paths, predictions):
    if prob >= 0.5:
        label = "Rotten"
        rotten_count += 1
    else:
        label = "Fresh"
        fresh_count += 1

    print(f"{path.name:30} -> {label:6} | Rotten score: {prob:.6f}")

print()
print("=" * 60)
print("RESULT")
print("=" * 60)
print(f"Total Rotten images : {len(paths)}")
print(f"Predicted Fresh     : {fresh_count}")
print(f"Predicted Rotten    : {rotten_count}")
print(f"Accuracy            : {rotten_count / len(paths) * 100:.2f}%")
