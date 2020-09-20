# Dependencies
import glob
import time
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Importing Training and Testing images using cv2
train_path = "data-science-bowl-2018\stage1_train"
all_trains = os.listdir(train_path)
test_path = "data-science-bowl-2018\stage1_test"
all_tests = os.listdir(test_path)
img_size = 128

train_img = []
for each_train in tqdm(all_trains):
    each_train_img = os.path.join(train_path, each_train, "images", each_train + ".png")
    each_train_img = cv2.imread(each_train_img, 1)
    each_train_img = cv2.resize(each_train_img, (img_size, img_size))
    train_img.append(each_train_img)
print('Done importing training images')
    
test_img = []
for each_test in tqdm(all_tests):
    each_test_img = os.path.join(test_path, each_test, "images", each_test + ".png")
    each_test_img = cv2.imread(each_test_img, 1)
    each_test_img = cv2.resize(each_test_img, (img_size, img_size))
    test_img.append(each_test_img)
print('Done importing testing images')

# Creating masks for the training set
masks_per_train = []
for each_train in tqdm(all_trains):
    mask_path = os.path.join(train_path, each_train, "masks")
    mask_names = os.listdir(mask_path)
    mask = np.zeros((img_size, img_size, 1))
    for each_mask in mask_names:
        each_mask_path = os.path.join(mask_path, each_mask)
        mask_image = cv2.imread(each_mask_path, -1)
        mask_image = cv2.resize(mask_image, (img_size, img_size))
        mask_image = np.expand_dims(mask_image, axis=-1)
        mask = np.maximum(mask, mask_image)
    masks_per_train.append(mask)


# HyperParameters
epochs = 10
batch_size = 8
filters = [64, 128, 256, 512, 1024]
kernel_size = (3,3)
input_shape = (img_size, img_size, 3)
optimizer = 'adam'
loss = "binary_crossentropy"

# Defining The Model
def UNet():
    inputs = keras.layers.Input((img_size, img_size, 3))
    
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


# Compiling the graphs to the model
model = UNet()
model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
model.summary()

# Training Step
model.fit(X_train, y_train, batch_size = 8, epochs = 5)

# Saving the weights
model.save_weights("weights/UNetW_1.h5")
# Loading weights
model.load_weights('weights/UNetW_1.h5')

# Importing test images
test2_path = "data-science-bowl-2018\stage2_test_final"
all_tests2 = os.listdir(test_path)

test2_img = []
for each_test in tqdm(all_tests2):
    each_test_img = os.path.join(test_path, each_test, "images", each_test + ".png")
    each_test_img = cv2.imread(each_test_img, 1)
    each_test_img = cv2.resize(each_test_img, (img_size, img_size))
    test2_img.append(each_test_img)
print('Done importing testing images')

# Predicting step
x = [np.array(test2_img[54])/255.0]
x = np.array(x)
print(x.shape)


result = model.predict(x)
result = result > 0.5


# Comparing the prediction with image
print(x.shape)
plt.imshow(x[0])
plt.title('my picture')
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255.0, (img_size, img_size)), cmap="gray")




















































































