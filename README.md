# Face-Mask-Detection
This project is implementation of Face Mask Detection using OpenCV and DeepLearning.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/70597091/122720047-99a65000-d28c-11eb-84db-8670e6b7e0e0.gif)

# Overview

![Screenshot from 2021-06-21 12-45-34](https://user-images.githubusercontent.com/70597091/122721873-c0658600-d28e-11eb-90ce-9ec200e6ba99.png)

The Detector works in two steps:
1. Haar Cascade Algorithm predicts the Region of Interest (ROI) for the current frame.
2. A CNN classifies the image in that (ROI) for Mask or No Mask Category.

# Requirements 
The code for this project was developed and tested on GPU (NVIDIA GTX 1650 4GB) powered laptop.

Some Dependencies used in the project are: 
1. OpenCV
2. Tensorflow v2.3
3. Python v3.8.5
4. Numpy v1.19.2
5. Matplotlib v3.3.2
6. Jupyter Notebook v6.2.0

# Traning
Haar Cascade Algorithm requires no training it is available [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).
The CNN requires training which is done by Transfer Learning on the pretrained Mobilenet V2 [model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2). The pretrained model was fintuned on [Face Mask Dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) available on Kaggle.

If you want to Train the CNN classifier on your custom dataset then open [model_training](https://github.com/HimGautam/Face-Mask-Detection/blob/main/model_training.ipynb) and import your own data of mask and non mask photos. Then Train the model, weights of this model will be saved as weights.hdf5 in the current folder. Then Proceed to Trying Step.

# Trying Out
If you have already trained the model in Training Step or Just want to test the face mask detector on the pretrained weights follow the steps below.
Open the file named [face_mask_detect](https://github.com/HimGautam/Face-Mask-Detection/blob/main/face_mask_detect.ipynb) with jupyter notebook and run all the cells. A new window will popup on the screen showing the output of this model.
