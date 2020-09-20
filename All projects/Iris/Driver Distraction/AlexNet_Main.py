import os
from glob import glob
import random
import time
import tensorflow as tf
import datetime
from PIL import Image
import PIL
import random
import numpy as np
from matplotlib.pyplot import imshow
import pandas as pd
from IPython.display import FileLink
import  matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
%matplotlib inline
from IPython.display import display, Image
import matplotlib.image as mpimg
import cv2
import time
import glob
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from tensorflow.python.keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from tensorflow import keras
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.applications.vgg16 import VGG16


# importing data from the csv
dataset = pd.read_csv('input_training/driver_imgs_list.csv')

by_drivers=dataset.groupby('subject')
unique_drivers = by_drivers.groups.keys()
# printing unique drivers
print(unique_drivers)

# HyperParameters
NUMBER_CLASSES = 10
img_rows = 224
img_cols = 224
color_type = 3
batch_size = 40
nb_epoch = 10


# Getting CV2 Image
def get_cv2_image(path, img_rows, img_cols, color_type=3):
    """
        description: To import image in an array
        
        input: 1. Path - path of the image to be extracted
               2. img_rows - width of the image after extraction
               3. img_cols - height of the image after extraction
               4. color_type - color scheme corresponding to opencv for extraction
            
        output: 1. The image in list format
    """
    # If the user want the image to be in grayscale
    if color_type == 1:
        # command for getting the image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # if the user want the image to be in rgb
    elif color_type == 3:
        # command for getting the image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    # resizing the image according to the arguments
    img = cv2.resize(img, (img_rows, img_cols)) 
    return img



# Load training data
def load_train(img_rows, img_cols, color_type=3):
    """
        description: Loading the training dataset
        
        input: 1. img_rows - width of the image after extraction
               2. img_cols - height of the image after extraction
               3. color_type - color scheme corresponding to opencv for extraction
            
        output: 1. Array of all the training iamges
                2. Label for the corresponging training image in the training dataset
    """
    # To keep track of how long it takes to import
    start_time = time.time()
    # To sotre the training images
    train_images = []
    # To store the lables for training images
    train_labels = []
    # for each class
    for classed in tqdm(range(NUMBER_CLASSES)):
        # Printing the class name(folder name)
        print('Loading directory c{}'.format(classed))
        # All the files(images) inside each folder(corresponding to each class)
        files = glob.glob(os.path.join('input_training', 'train', 'c' + str(classed), '*.jpg'))
        # looping over all files
        for file in files:
            # importing using the above function
            img = get_cv2_image(file, img_rows, img_cols, color_type)
            # appending the imported img into the train_images
            train_images.append(img)
            # appending the class for each image in teh train_images
            train_labels.append(classed)
    # Total time taken to complete
    print("Data Loaded in {} second".format(time.time() - start_time))
    return train_images, train_labels


# Normalizing and preprocessing training data
def read_and_normalize_train_data(img_rows, img_cols, color_type):
    """
        description: Normalizing the train data, splitting and getting it ready for training
                     One-hot encoding of labels
        
        input: 1. img_rows - width of the image after extraction
               2. img_cols - height of the image after extraction
               3. color_type - color scheme corresponding to opencv for extraction
            
        output: 1. x_train
                2. x_test
                3. y_train
                4. y_test
    """
    # Loading all the training data into X and labels
    X, labels = load_train(img_rows, img_cols, color_type)
    # Encoding labels into one-hot
    y = np_utils.to_categorical(labels, 10)
    # Splitting into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Formatting as input for the Network
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1,img_rows,img_cols,color_type)
    x_test = np.array(x_test, dtype=np.uint8).reshape(-1,img_rows,img_cols,color_type)
    
    return x_train, x_test, y_train, y_test


# Loading Testing data
def load_test(size=200000, img_rows=64, img_cols=64, color_type=3):
    """
        description: Importing the data for testing
        
        input: 1. size = Maximum number of testing images
               2. img_rows - width of the image after extraction
               3. img_cols - height of the image after extraction
               4. color_type - color scheme corresponding to opencv for extraction
            
        output: 1. x_test
                2. corresponding label to the above test set
    """
    # Path
    path = os.path.join('input_training', 'test', '*.jpg')
    # All the files inside the path
    files = sorted(glob.glob(path))
    # Initiating X_test and X_test_id for storing the images and corresponding labels
    X_test, X_test_id = [], []
    # Keeping count of total number of images
    total = 0
    # Number of files in the path
    files_size = len(files)
    # Looping over all the files
    for file in tqdm(files):
        # If total is greater than the max number of files
        if total >= size or total >= files_size:
            break
        # File name
        file_base = os.path.basename(file)
        # Getting the image in an array
        img = get_cv2_image(file, img_rows, img_cols, color_type)
        # Appending it to the X_test
        X_test.append(img)
        # Appending the corresponding label
        X_test_id.append(file_base)
        # Increment total by 1
        total += 1
    return X_test, X_test_id

# Normalizing and preprocessing testing data
def read_and_normalize_sampled_test_data(size, img_rows, img_cols, color_type=3):
    """
        description: Normalizing the test data, splitting and getting it ready for training
        
        input: 1. size - max number of images allowed to be extracted
               2. img_rows - width of the image after extraction
               3. img_cols - height of the image after extraction
               4. color_type - color scheme corresponding to opencv for extraction
            
        output: 1. x_test
                2. corresponding classes
    """
    # Get test_data and test_id
    test_data, test_ids = load_test(size, img_rows, img_cols, color_type)
    
    # Changing the data type to uint8
    test_data = np.array(test_data, dtype=np.uint8)
    # Reshaping to given size
    test_data = test_data.reshape(-1,img_rows,img_cols,color_type)
    
    return test_data, test_ids


# Getting Test and Train
x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_rows, img_cols, color_type)
print('Train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
# number of test samples we decide to check teh accuracy of the model
nb_test_samples = 200
# Getting test_files and the corresponding labels
test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols, color_type)
print('Test shape:', test_files.shape)
print(test_files.shape[0], 'Test samples')


# Data Summary
# Name of classes
names = [item[17:19] for item in sorted(glob.glob("input/train/*/"))]
names = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
# Number of files for testing
test_files_size = len(np.array(glob.glob(os.path.join('input_training', 'test', '*.jpg'))))
# Number of images for training
x_train_size = len(x_train)
# Number of classes
categories_size = len(names)
# Number of validation images
x_test_size = len(x_test)
print('There are %s total images.\n' % (test_files_size + x_train_size + x_test_size))
print('There are %d training images.' % x_train_size)
print('There are %d total training categories.' % categories_size)
print('There are %d validation images.' % x_test_size)
print('There are %d test images.'% test_files_size)


# Class mapped with name of class
activity_map = {'c0': 'Safe driving', 
                'c1': 'Right - Texting', 
                'c2': 'Right - Talking on the phone', 
                'c3': 'Left - Texting', 
                'c4': 'Left - Talking on the phone', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}


# EarlyStopping
# Save model to
models_dir = "saved_models"
# Make directory if there is no directory
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
# Checkpointer
checkpointer = ModelCheckpoint(filepath='saved_models/weights_best_vanilla.hdf5', 
                               monitor='val_loss', mode='min',
                               verbose=1, save_best_only=True)
# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
# Callbacks
callbacks = [checkpointer, es]


# Model Architecture
def create_model_v1():
    """
        description: Defining the Model with keras backend
            
        output: 1. The Model
    """
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(img_rows, img_cols, color_type), kernel_size=(11,11),strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    return model

def get_f1(y_true, y_pred): #taken from old keras source code
    """
        description: F1 value for accuracy
        
        input:  1. Real y to compare from
                2. Predicted y to check accuracy from
        
        output: 1. The F1 value
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Model Summary
# Create model
model_v1 = create_model_v1()
# Model.summary from keras
model_v1.summary()
# Settign optimizer, loss and metrics for the model
model_v1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Load weights from previouslt trained models
model_v1.load_weights('saved_models/weights_best_vanilla.hdf5')

# MODEL EVALUATION
def plot_train_history(history):
    """
        description: Model Evaluation
        
        input: 1. Weigths after fitting the model
        
        output: 1. Accuray and loss over validation set
    """
    # Plotting acc form keras
    plt.plot(history.history['acc'])
    # Plotting val_acc from keras
    plt.plot(history.history['val_acc'])
    # Title of the plot
    plt.title('Model accuracy')
    # ylabel for the plot
    plt.ylabel('accuracy')
    # Xlabel for the plot
    plt.xlabel('epoch')
    # Legends for the plot
    plt.legend(['train', 'test'], loc='upper left')
    # Command to display the plot
    plt.show()

    # Plotting the loss from keras
    plt.plot(history.history['loss'])
    # Plotting the val_loss from keras
    plt.plot(history.history['val_loss'])
    # Title of the plot
    plt.title('Model loss')
    # ylabel for the plot
    plt.ylabel('loss')
    # Xlabel for the plot
    plt.xlabel('epoch')
    # Legends for the plot
    plt.legend(['train', 'test'], loc='upper left')
    # Command to display the plot
    plt.show()

plot_train_history(history_v1)

# Scoring the model with the test set
score = model_v1.evaluate(x_test, y_test, batch_size=32)

# Accuracy and Loss
# Printing Loss and Accuracy
print('Loss     = ' ,score[0])
print('Accuracy = ' ,score[1])
