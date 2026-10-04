from pathlib import Path

import numpy as np
import tensorflow as tf

from PIL import Image


MODEL_PATH = Path("models/custom_cnn_best.keras")

model = tf.keras.models.load_model(
    MODEL_PATH
)

image_path = input("Enter image path: ").strip()

image = Image.open(
    image_path
).convert("RGB")

image = image.resize(
    (160, 160)
)

image_array = np.array(
    image
).astype("float32") / 255.0

image_array = np.expand_dims(
    image_array,
    axis=0
)

prediction = float(
    model.predict(
        image_array,
        verbose=0
    )[0][0]
)

print()
print("=" * 50)
print("Custom CNN Prediction")
print("=" * 50)

print(f"Raw probability: {prediction:.10f}")

if prediction >= 0.5:
    print("Prediction: Rotten")
    print(f"Confidence: {prediction * 100:.2f}%")
else:
    print("Prediction: Fresh")
    print(f"Confidence: {(1 - prediction) * 100:.2f}%")
