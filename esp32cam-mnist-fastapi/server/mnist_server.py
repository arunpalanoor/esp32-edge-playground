import logging
import os

# Must be set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

app = FastAPI()

# Load MNIST model (Keras .h5)
model = tf.keras.models.load_model("../model/model.h5")

# Request schema
class MNISTImage(BaseModel):
    pixels: list  # 784 integers (0–255)

@app.post("/predict")
def predict_digit(data: MNISTImage):
    #Plot data received as image by reshaping to 28x28 and displaying as image
    # img = np.array(data.pixels, dtype=np.float32).reshape(28, 28)
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # Convert list → numpy array
    arr = np.array(data.pixels, dtype=np.float32)

    # Normalize to 0–1
    arr = arr / 255.0

    # Reshape to (1, 28, 28, 1)
    arr = arr.reshape(1, 784)

    # Run inference
    preds = model.predict(arr)
    digit = int(np.argmax(preds))
    print(f"Predicted digit: {digit}")

    return {"digit": digit}