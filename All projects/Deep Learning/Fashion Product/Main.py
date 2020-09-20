#https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/notebooks

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import gc

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
import swifter

from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

from sklearn.metrics.pairwise import pairwise_distances

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred,axis = 1) 
    y_test_classes = np.argmax(y_test, axis = 1)
    cm = confusion_matrix(y_test_classes, y_pred_classes) 
    sns.heatmap(cm, annot = True,fmt='.0f')
    plt.show()
    
def load_images(names, articletype):
    image_array = []
    for image_name in tqdm(names, desc = 'reading images for ' + articletype):
        img_path = IMAGE_PATH + image_name
        try:
            img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        except:
            continue
        img = image.img_to_array(img)
        image_array.append(img)
    return np.array(image_array)

def get_recommender(idx, df, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_similarity[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim

DATA_PATH = 'fashion-dataset/'
IMAGE_PATH = DATA_PATH + 'images/'
IMAGE_SIZE = 224
LIMIT_IMAGES = 1000

dfstyles = pd.read_csv(DATA_PATH + 'styles.csv', error_bad_lines=False, warn_bad_lines=False)
dfstyles['image'] = dfstyles['id'].map(lambda x: str(x) + '.jpg')
dfstyles.columns = dfstyles.columns.str.lower()
print(dfstyles.shape)
dfstyles.head()

dfstyles['articletype'].nunique()
dfstyles['articletype'].value_counts().head()

dfstyles['cntarticle'] = dfstyles.groupby('articletype')['id'].transform('count')
dfdata = dfstyles[dfstyles['cntarticle'] > 500]
print(dfdata.shape, dfdata['articletype'].nunique())
dfarticles =dfdata.groupby('articletype',as_index=False)['id'].count()
dfarticles

imglist = [IMAGE_PATH + x for x in dfdata['image'].sample(10).values]

fig,ax = plt.subplots(2,5,figsize=(18,10))
for index, img_file in enumerate(imglist):
    img = plt.imread(img_file)
    x = int(index / 5)
    y = index % 5
    ax[x,y].imshow(img)
plt.show()  

image_list = []
article_list = []
for index, grouprow in dfarticles.iterrows():
    if index > 4:
        continue
    image_names = dfdata[dfdata['articletype'] == grouprow['articletype']]['image'].values
    if len(image_names) > LIMIT_IMAGES:
        image_names = image_names[:LIMIT_IMAGES]
    image_list.extend(load_images(image_names, grouprow['articletype']))
    article_list.extend(len(image_names) * [grouprow['articletype']])
    
X = np.array(image_list) / 255.0
X = X.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
y_encoded = LabelEncoder().fit_transform(article_list)
print("Number of classes : ",np.unique(y_encoded, return_counts=True))
y = to_categorical(y_encoded, num_classes = len(np.unique(article_list)))
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_test.shape)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (224,224,3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(article_list)), activation = "softmax"))
print(model.summary())

del X,y,article_list,image_list
gc.collect()

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

EPOCHS = 3
BATCH_SIZE = 64

datagen = image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE), epochs = EPOCHS, 
                              validation_data = (X_test,y_test), verbose = 2, steps_per_epoch=X_train.shape[0] // BATCH_SIZE
                              , callbacks=[learning_rate_reduction])

plot_confusion_matrix(model, X_test, y_test)
