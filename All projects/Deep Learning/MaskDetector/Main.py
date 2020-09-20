import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import glob
import cv2

from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.utils import shuffle

activity_map = {'c0': 'Mask', 
                'c1': 'No_mask'}

# Importing the data
train_mask_path = glob.glob('data_split/train/with_mask/*.jpg')
train_without_mask_path = glob.glob('data_split/train/without_mask/*.jpg')
test_mask_path = glob.glob('data_split/test/with_mask/*.jpg')
test_without_mask_path = glob.glob('data_split/test/without_mask/*.jpg')
val_mask_path = glob.glob('data_split/val/with_mask/*.jpg')
val_without_mask_path = glob.glob('data_split/val/without_mask/*.jpg')


# Analysing the data
print(len(train_mask_path))
print(len(train_without_mask_path))
print(len(test_mask_path))
print(len(test_without_mask_path))
print(len(val_mask_path))
print(len(val_without_mask_path))

# Preparing holders for the Data
X_train = []
X_test = []
X_val = []
y_train = []
y_test = []
y_val = []

# Changing jpg files into numpy, resizing and normalizing
for imgs in train_mask_path:
    img = cv2.imread(imgs, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    X_train.append(img)
for imgs in train_without_mask_path:
    img = cv2.imread(imgs, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    X_train.append(img)

for imgs in test_mask_path:
    img = cv2.imread(imgs, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    X_test.append(img)
for imgs in test_without_mask_path:
    img = cv2.imread(imgs, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    X_test.append(img)
    
for imgs in val_mask_path:
    img = cv2.imread(imgs, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    X_val.append(img)
for imgs in val_without_mask_path:
    img = cv2.imread(imgs, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    X_val.append(img)
    
    
# Creating labels
for i in range(len(train_mask_path)):
    y_train.append(1)
for i in range(len(train_without_mask_path)):
    y_train.append(0)
    
for i in range(len(test_mask_path)):
    y_test.append(1)
for i in range(len(test_without_mask_path)):
    y_test.append(0)

for i in range(len(val_mask_path)):
    y_val.append(1)
for i in range(len(val_without_mask_path)):
    y_val.append(0)
    
    
print(len(X_train))
print(len(X_test))
print(len(X_val))
print(len(y_train))
print(len(y_test))
print(len(y_val))
    
    
# Shuffling the data since both classes were clumped together
X_train, y_train = shuffle(X_train, y_train, random_state=1)
X_test, y_test = shuffle(X_test, y_test, random_state=4)
X_val, y_val = shuffle(X_val, y_val, random_state=12)

# Data Augmentation
datagen = ImageDataGenerator(featurewise_center=True, 
                             featurewise_std_normalization=True, 
                             zca_whitening=True,
                             rotation_range=90,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             )

# Saving the data in npy format
np.save('X_train', X_train)
np.save('X_test', X_test)
np.save('X_val', X_val)
np.save('y_train', y_train)
np.save('y_test', y_test)
np.save('y_val', y_val)

# OneHot encoding
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)
y_val = np_utils.to_categorical(y_val, 2)


# Reshaping to fit into the model
X_train = np.array(X_train, dtype=np.uint8).reshape(-1, 32, 32, 3)
X_test = np.array(X_test, dtype=np.uint8).reshape(-1, 32, 32, 3)


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(32, 32, 3))
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(2, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model = create_res_net()

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    verbose=1,
    validation_data=(X_test, y_test),
    batch_size=128,
)


# Plotting the model accuracu
def plot_train_history(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
plot_train_history(history)


# Preparing the model for manual input
num = 12
img = cv2.imread(test_mask_path[num], cv2.IMREAD_COLOR)
pred_img = cv2.resize(img, (32, 32))
pred_img = np.array(pred_img, dtype=np.uint8).reshape(-1, 32, 32, 3)
y_prediction = model.predict(pred_img, verbose=1)
y_pred = format(activity_map.get('c{}'.format(np.argmax(y_prediction))))
print(y_pred)
print(y_prediction)
    

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    