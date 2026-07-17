# Import the necessary libraries.
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Import the MNIST data set.
from keras.datasets import mnist

# Load and preprocess the data.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalise the pixel values to be between 0 and 1.
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# View output.
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

# Flatten the images.
X_train = X_train.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

# View output.
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

# One-hot encode the labels.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# View the output.
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# Split the training data into training and validation sets.
# 50,000 samples for training, rest for validation
split = 50000

x_val = X_train[split:]
y_val = y_train[split:]
X_train = X_train[:split]
y_train = y_train[:split]

# View output.
print("x_val shape: ", x_val.shape)
print("y_val shape: ", y_val.shape)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
