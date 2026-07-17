import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

# Load MNIST model (Keras .h5)
model = tf.keras.models.load_model("model.h5")

# Request schema
class MNISTImage(BaseModel):
    pixels: list  # 784 integers (0–255)

@app.post("/predict")
def predict_digit(data: MNISTImage):
    # Convert list → numpy array
    arr = np.array(data.pixels, dtype=np.float32)

    # Normalize to 0–1
    arr = arr / 255.0

    # Reshape to (1, 28, 28, 1)
    arr = arr.reshape(1, 28, 28, 1)

    # Run inference
    preds = model.predict(arr)
    digit = int(np.argmax(preds))

    return {"digit": digit}
