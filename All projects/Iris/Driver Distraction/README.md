# Driver Distraction
  An algorithm to find if the driver of a vehicle is distracted.

## Objective
  Creating a robust model which works in every environment and during any time of the day

## Requirements
  The code is in [Python 3.7]('https://www.python.org/downloads/release/python-379/')</br>
  The dependencies are listed on Main.py 

## Abstract
  <p>According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver. This roughly translates to 425,000 people injured and 3,000 people killed by distracted driving every year.</p>
  <p>We created an system which can identify and send an alert if the driver is engaged in any other activity (among the 10 activities recorded in the dataset).</p>
  
## Data Set
  Data is obtained from Kaggle “[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)”, and contains images of people involved in one of these 9 types of distractions. The volunteers are chosen from different race and ethnicity, to make the model created with this dataset much more robust. The train and test data are split on the drivers, such that one driver can only appear either train or test set.</br>
### Classes
  <ul>
  <li>Safe Driving</li>
  <li>Texting - Right</li>
  <li>Texting - Left</li>
  <li>Talking on phone - Right</li>
  <li>Talking on phone - Left</li>
  <li>Operating the radio</li>
  <li>Drinking</li>
  <li>Reaching behind</li>
  <li>Hair and Makeup</li>
  <li>Talking to passenger</li>
  </ul>
  
  
## The Architectures
  We tried 3 different architectures,
  <ul>
  <li>A Vanilla CNN</li>
  <li>AlexNet</li>
  <li>VGG16</li>
  </ul>
  
  ### Vanilla CNN
   <p>The model has 4 convolutional layers and a max pooling after each convolutional layer for down-sampling and a dropout of 0.5 is applied after the convolutional layers and one just before the output layer</p>
   <p>The training dataset is sent in 32 pictures per batch for 10 epochs and set to stop when the validation accuracy stops imporving (which almost always happens at 3rd or 4th epoch).</p>
   <p>RMS Prop is used as the optimizer with Categorical cross entropy as the loss function.</p>
   <p>The validation accuracy for this model was 91% which is really good but the model didn't perform good enough for any input which didn't resemble the original dataset</p>
  <img src='https://github.com/Revanthmk/Proofs/blob/master/Pictures%20proof/Driver%20Distraction/CNN%20visu.PNG'>
  
  ### AlexNet
   <p>This is using the architecutre AlexNet (Not using a pretrained model).</p>
   <p>The data is resized to 32x32 and sent in batches of 100 with default learning rate of 0.001 as recommended in the AlexNet research paper.</p>
   <p>The loss function used was Categorical Crossentropy with Adam optimizer.</p>
   <p>The only change made in the architecture was in the outputlayer, changing it from 1000 nodes to 10 nodes to fit the use case. More changes can be mode to make the architecture more efficient for this specific dataset but I'm not experienced enough to make changes in any architectures yet.</p>
   <p>The model didn't do as accurate as the previous model with 87% on validation set but did significantly better for rela life input cases, which shows that the model which we previously used was overfitted.</p>
   <img src='https://github.com/Revanthmk/Proofs/blob/master/Pictures%20proof/Driver%20Distraction/AlexNet.png'>
   
  ### VGG16
   <p>This is not using the architecture VGG16 but the pretrained model on ImageNet.</p>
   <p>Every layer except the last 2 were frozen because those contains complex features extracted from the input image.</p>
   <p>Added 2 dense layers with 1000 nodes each and an output layer with 10 nodes.</p>
   <p>We choose VGG16 over VGG19 the model is deployed on a remote device with very limited processing power, and VGG16 is almost as good as VGG19 which isn't worth the extra processing power required.</p>
   <p>The loss function used was Categorical Crossentropy with RMSprop as the oprimizer.</p>
   <p>The model had accuracy of 78% which shows that it avoided the overfitting problem with our Vanilla CNN model but AlexNet had a better accuracy so we decided to go with that model.</p>
   <p>The low accuracy of the model might be because of the pretrained weight which was vastly different from our dataset.</p>
   <img src='https://github.com/Revanthmk/Proofs/blob/master/Pictures%20proof/Driver%20Distraction/VGG16.PNG'>
   
   
   
   
























