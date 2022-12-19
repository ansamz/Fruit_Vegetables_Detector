# Fruits and Vegetables Detector


![alternative text](./app_styling/f_g.jpg)


## Project Intro/Objective
The purpose of this project is to write an algorithm to detect fruits and vegetables.

## Project Description
First I trained a Convolutional Neural Network on the dataset, it performs rather well on the test data with 99.20% accuracy. Regarding transfer learning using VGG-16 94.01% accuracy on test data and 98.07% accuracy on the MobileNetV2.

I also used TensorFlow Object Detection API to detect multiple fruits and vegetables in one picture, this is an opensource computer vision framework for building object detection and image segmentation models that can localize multiple objects in the same image, here I used VGG16, ResNet50, and EfficientNetB0 (not the best performance).

A streamlit app was created so you can test images on various models.

Note:
VGG-16 was not included in the streamlit app because the saved model was too big but you can find it in the [notebook](notebooks/Fruit_vegetables_detection_final.ipynb).

Object Detection API citation:
 https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/


### Streamlit App Link:
https://ansamz-fruit-vegetables-detector-fruit-veg-app-2jocy1.streamlit.app/


### Methods Used
* Deep Learning
* Data Visualization
* Predictive Modeling
* Transfer Learning

### Technologies
* tensorflow
* Python
* Pandas, jupyter
* Streamlit
* Google Colab Pro
* pillow
* Object Detection API

### Data
Download the whole dataset from Kaggle: https://www.kaggle.com/datasets/moltean/fruits

This repo contains a few  samples from the data.

## Models
* VGG16 (Object Detection API)
* ResNet50 (Object Detection API)
* EfficientNetB0 (Object Detection API)
* CNN
* MobileNetV2
* VGG-16(transfer learning on dataset)


## Folders and Files:
* app_styling: contains the picture added to the streamlit app.
* cnn_model and mobilenet_model: save folders for the trained models.
* data: contains samples of saved data from the data I used to train and test the models.
* notebooks: 
    - Fruit_vegetables_detection_final.ipynb [notebook](notebooks/Fruit_vegetables_detection_final.ipynb): The code where I trained the CNN, VGG-16 and MobileNetV2 models.
    - Object_Detection_API.ipynb [notebook](notebooks/Object_Detection_API.ipynb): the code where I implemented Object detection API
* fruit_veg_app.py [notebook](fruit_veg_app.py): streamlit app code

## TODO
YOLO
You Only Look Once:
Unified, Real-Time Object Detection
Citation: 
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf

The pictures need labeling before training the model and it takes time.