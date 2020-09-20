import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import accuracy_score
import os
import scipy.io
import math

from sklearn.utils import shuffle

from PIL import Image
import requests
from io import BytesIO

from tensorflow.keras.applications import resnet50
from keras.preprocessing import image

import matplotlib.pyplot as plt

import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions

import tensorflow as tf
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

from scipy import spatial
from tqdm import tqdm

import gc

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils



# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Reshaping the input to fit the model
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# OneHot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Build model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))















