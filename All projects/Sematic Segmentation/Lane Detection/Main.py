import glob
import time
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import time
import pickle

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model


# Importing the data from pickle
image_dataset = open('Dataset/Lane detection/full_CNN_train.p','rb')
image_dataset = pickle.load(image_dataset)
image_dataset = np.array(image_dataset)/255.0

label_dataset = open('Dataset/Lane detection/full_CNN_labels.p','rb')
label_dataset = pickle.load(label_dataset)
label_dataset = np.array(label_dataset)/255.0

# Creating the mask
a = np.zeros(label_dataset.shape, dtype=float32)
label_dataset = np.concatenate((a, label_dataset, a), axis=3)

# Creating the X and y train
X_train = image_dataset
y_train = label_dataset

# Showing the Image
plt.axis('off')
plt.imshow(np.array(image_dataset[10075]))
# Showing the mask
plt.imshow(np.array(label_dataset[10075]))

# Checking the size
image_dataset.shape
label_dataset.shape

# HyperParameters
epochs = 10
batch_size = 8
filters = [64, 128, 256, 512, 1024]
kernel_size = (3,3)
img_height = 80
img_width = 160
optimizer = 'adam'
loss = "binary_crossentropy"

# The Model
def UNet():
    inputs = keras.layers.Input((img_height, img_width, 3))
    
    l0 = inputs
    c0 = Conv2D(filters=filters[0], kernel_size=kernel_size, padding='same', activation='relu')(l0)
    c0 = Conv2D(filters=filters[0], kernel_size=kernel_size, padding='same', activation='relu')(c0)
    p0 = MaxPooling2D(pool_size=(2,2), padding="same")(c0)

    c1 = Conv2D(filters=filters[1], kernel_size=kernel_size, padding='same', activation='relu')(p0)
    c1 = Conv2D(filters=filters[1], kernel_size=kernel_size, padding='same', activation='relu')(c1)
    p1 = MaxPooling2D(pool_size=(2,2), padding="same")(c1)
    
    c2 = Conv2D(filters=filters[2], kernel_size=kernel_size, padding='same', activation='relu')(p1)
    c2 = Conv2D(filters=filters[2], kernel_size=kernel_size, padding='same', activation='relu')(c2)
    p2 = MaxPooling2D(pool_size=(2,2), padding="same")(c2)
    
    c3 = Conv2D(filters=filters[3], kernel_size=kernel_size, padding='same', activation='relu')(p2)
    c3 = Conv2D(filters=filters[3], kernel_size=kernel_size, padding='same', activation='relu')(c3)
    p3 = MaxPooling2D(pool_size=(2,2), padding="same")(c3)
    
    b0 = Conv2D(filters=filters[4], kernel_size=kernel_size, padding='same', activation="relu")(p3)
    b0 = Conv2D(filters=filters[4], kernel_size=kernel_size, padding='same', activation="relu")(b0)
    
    u0 = UpSampling2D((2, 2))(b0)
    co0 = Concatenate()([u0, c3])
    c4 = Conv2D(filters=filters[3], kernel_size=kernel_size, padding='same', activation="relu")(co0)
    c4 = Conv2D(filters=filters[3], kernel_size=kernel_size, padding='same', activation="relu")(c4)
    
    u1 = UpSampling2D((2, 2))(c4)
    co1 = Concatenate()([u1, c2])
    c5 = Conv2D(filters=filters[2], kernel_size=kernel_size, padding='same', activation="relu")(co1)
    c5 = Conv2D(filters=filters[2], kernel_size=kernel_size, padding='same', activation="relu")(c5)
    
    u2 = UpSampling2D((2, 2))(c5)
    co2 = Concatenate()([u2, c1])
    c6 = Conv2D(filters=filters[1], kernel_size=kernel_size, padding='same', activation="relu")(co2)
    c6 = Conv2D(filters=filters[1], kernel_size=kernel_size, padding='same', activation="relu")(c6)
    
    u3 = UpSampling2D((2, 2))(c6)
    co3 = Concatenate()([u3, c0])
    c7 = Conv2D(filters=filters[0], kernel_size=kernel_size, padding='same', activation="relu")(co3)
    c7 = Conv2D(filters=filters[0], kernel_size=kernel_size, padding='same', activation="relu")(c7)
    
    outputs = Conv2D(1,(1,1), activation='sigmoid')(c7)
    model = Model(inputs, outputs)
    return model


# Model Initiating
model = UNet()
model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
model.summary()

# The training step
model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)

# Saving and loading weights
model.save_weights("weights/UNetW_1.h5")
model.load_weights('weights/UNetW_1.h5')

# Prediction step
x = [np.array(image_dataset[10075])]
x = np.array(x)
print(x.shape)
result = model.predict(x)
result = result > 0.5

# Printing the prediction
print(x.shape)
plt.imshow(x[0])
plt.title('my picture')
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255.0, (img_height, img_width)), cmap="gray")

# Using outside image from prediction
img = cv2.imread('test/8.jpg', 3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img_width, img_height))

x = [np.array(img)]
x = np.array(x)
print(x.shape)


result = model.predict(x)
result = result > 0.5

print(x.shape)
plt.imshow(x[0])
plt.title('my picture')
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255.0, (img_height, img_width)), cmap="gray")
































