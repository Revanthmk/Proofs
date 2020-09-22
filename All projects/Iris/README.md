# Driver Distraction
  An algorithm to find if the driver of a vehicle is distracted.

## Objective
  Creating a robust model which works in every environment and during any time of the day

## Requirements
  The code is in Python 3.7</br>
  All the dependencies can be installed using requirement.txt

## Abstract
  <p>According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver. This roughly translates to 425,000 people injured and 3,000 people killed by distracted driving every year.</p>
  <p>We created an system which can identify and send an alert if the driver is engaged in any other activity(among the 10 activities recorded in the dataset).</p>
  
## The model Architecture
  We tried 3 different architectures,
  <ul>
  <li>A Vanilla CNN</li>
  <li>AlexNet</li>
  <li>VGG16</li>
  </ul>
  
  ### Vanilla CNN
   The model has 4 convolutional layers and a max pooling after each convolutional layer for down-sampling and a dropout of 0.5 is applied after the convolutional layers and one just before the output layer</br>
   The training dataset is sent in 32 pictures per batch for 10 epochs and set to stop when the validation accuracy stops imporving(which almost always happens at 3rd or 4th epoch).</br>
   RMS Prop is used as the optimizer with Categorical cross entropy as the loss function.</br>

  <img src='https://github.com/Revanthmk/Proofs/blob/master/Pictures%20proof/Driver%20Distraction/CNN%20visu.PNG'>
  (not actual representation of the [model])




















