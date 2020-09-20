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


### Model Architecture
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

# Load weights from previouslt trained models
model_v1.load_weights('saved_models/weights_best_vanilla.hdf5')


# Single Input
def plot_test_classs(model, manual_imgs, image_number, color_type=3):
     """
        description: Predicting from raw input after preprocessing to the corresponding model
        
        input:  1. Model
                2. Images to be predicted
                3. Index of image from the set of Images
                4. Color type of the image
        
        output: 1. Prediction
    """
    # Selecting a specific image from the set of images sent for prediction
    img_brute = manual_imgs[image_number]
    # Resizing it to fit the model
    img_brute = cv2.resize(img_brute,(img_rows,img_cols))
    # Showing the image
    plt.imshow(img_brute, cmap='gray')
    
    # Resing it to fit the model
    new_img = img_brute.reshape(-1,img_rows,img_cols,color_type)
    # Prediction
    y_prediction = model.predict(new_img, batch_size=batch_size, verbose=1)
    # Mapping prediction with activity map
    predicted_text = format(activity_map.get('c{}'.format(np.argmax(y_prediction))))
    
    # Getting image to write over
    img_dis = get_cv2_image(manual_pics[image_number],640,480)
    # Changing color code from BGR to RGB
    img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)
    # Showing the image
    plt.imshow(img_dis)
    cv2.imshow('test',img_dis)
    # Writing the prediction over the image
    write_img(image_number, predicted_text)
    # Printing the predicted text
    print(predicted_text)
    
plot_test_classs(model_v1, manual_imgs, 0)



# Capturing framewise
def write_img(image_number, text):
    """
        description: Writing the given text over a picture
        
        input:  1. Index of the image
                2. Text
        
        output: 1. The image after writing over it
    """
    # Spacial position of the text
    org = (30,450)
    # Font scale
    fontscale = 1.5
    # Color of the text
    color = (255,0,0)
    # Thickness of the font
    thickness=2
    # Font
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # PutText command form openCV
    imagee = cv2.putText(manual_displays[image_number], text, org, font,  
                       fontscale, color, thickness, cv2.LINE_AA)
    # Changing color from BGR to RGB
    imagee = cv2.cvtColor(imagee, cv2.COLOR_BGR2RGB)
    # Showing the image
    plt.imshow(imagee)
    cv2.imshow('test',imagee)
    return imagee


def import_images(path = 'prediction/input/framewise/*.jpg'):
    # Path where the image is stored
    path = 'C:\\Users\\revan\\JUPYTERRRRRRRRRRRRRRRRRR\\Iris\\Drowsiness\\input_training\\train\\c4\\img_803.jpg'
    
    manual_pics = glob.glob(path)
    
    manual_imgs = []
    # Array for storing all the raw pictures
    manual_displays = []
    
    for file in manual_pics:    
        
        manual_display = get_cv2_image(file,640,480)
       
        manual_displays.append(manual_display)
        # For prediction
        manual = get_cv2_image(file, img_rows=224, img_cols=224, color_type=3)
        # For flipping the image
        #manual = img_flip_lr = cv2.flip(manual, 1)
        # Appending prediction image into manual_imgs
        manual_imgs.append(manual)

    # Changing into uint8
    manual_imgs = np.array(manual_imgs, dtype=np.uint8)
    # Changing into uint8
    manual_imgs = manual_imgs.reshape(-1,img_rows,img_cols,3)
    
    
    
    # Path where the image is stored
    path = 'prediction/input/framewise/*.jpg'
    j = 0
    # Set of all pixtures in the path
    manual_pics = sorted(glob.glob(path), key=os.path.getmtime)
    # Array for storing all the pictures after preprocessing
    manual_imgs = []
    # Array for storing all the raw pictures
    manual_displays = []
    # Looping over all pictures in the path
    for file in manual_pics:
        # For displayingn the picture and writing over it
        manual_display = get_cv2_image(file,640,480)
        # For flipping the image
        # manual_display = img_flip_lr = cv2.flip(manual_display, 1)
        # Appending display image into manual_display
        manual_displays.append(manual_display)
        manual = get_cv2_image(file, img_rows=224, img_cols=224, color_type=3)

        manual_imgs.append(manual)
        j += 1

    manual_imgs = np.array(manual_imgs, dtype=np.uint8)
    manual_imgs = manual_imgs.reshape(-1,img_rows,img_cols,3)
    return manual_imgs, manual_displays

def plot_test_classs(model, manual_imgs, image_number, color_type=3):
    img_brute = manual_imgs[image_number]
    img_brute = cv2.resize(img_brute,(224,224))
    plt.imshow(img_brute, cmap='gray')

    new_img = img_brute.reshape(-1,224,224,3)
    y_prediction = model.predict(new_img, batch_size=batch_size, verbose=1)
    predicted_text = format(activity_map.get('c{}'.format(np.argmax(y_prediction))))
    
    #displaying picture
    editted = write_img(image_number, predicted_text)
    write_img(image_number, predicted_text)
    editted = cv2.cvtColor(editted, cv2.COLOR_BGR2RGB)
    name = i
    cv2.imwrite('prediction/output/images/%s.jpg' %name, editted)


def get_video():
    video = cv2.VideoCapture('prediction/input/project.mp4');

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)

    img_array = []
    for filename in sorted(glob.glob('prediction/output/images/*.jpg'), key=os.path.getmtime):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        img_array.append(img)
    size = (width,height)

    out = cv2.VideoWriter('prediction/output/Video/outfvput.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    
FrameCapture('prediction/input/VID20200504144113-1.mp4')

list = os.listdir("C:\\Users\\revan\\JUPYTERRRRRRRRRRRRRRRRRR\\Iris\Drowsiness\\prediction\\input\\framewise")
number_files = len(list)
for i in range(number_files):
    plot_test_classs(model_v1, manual_imgs, i)

get_video()