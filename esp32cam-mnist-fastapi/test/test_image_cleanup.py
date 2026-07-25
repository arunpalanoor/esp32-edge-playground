import os
import numpy as np
import matplotlib.pyplot as plt

import cv2

import requests
import tensorflow as tf

# Load original image
image_path = "../server/original.png"
org_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Display image function
def show_image(image_name):
    plt.imshow(image_name, cmap ='gray')
    plt.show()

# Convert the image to black and white
black_white_image = cv2.threshold(org_image, 127, 255, cv2.THRESH_BINARY)[1]
show_image(black_white_image)

# THRESH_BINARY makes dark ink = 0, so invert first so digit pixels are non-zero
coords = cv2.findNonZero(cv2.bitwise_not(black_white_image))
if coords is None:
    print("No dark pixels found")
else:
    x, y, w, h = cv2.boundingRect(coords)
    padding = 30
    cropped_image = black_white_image[y-padding:y+h+padding, x-padding:x+w+padding]
    blurred = cv2.GaussianBlur(cropped_image, (3, 3), 0)
    #show_image(blurred)

resized = cv2.resize(blurred, (28, 28), interpolation=cv2.INTER_AREA)

inverted_image = 255 - resized

show_image(inverted_image)

# Load MNIST model
model = tf.keras.models.load_model("../model/model.h5")

# Send image to server
array = np.array(inverted_image)
#print(array.shape)
array = array.reshape(1, 784)
array = array.astype(np.float32) / 255.0

prediction = model.predict(array)
print(np.argmax(prediction))