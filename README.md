# Fruits and Vegetables Detector


![alternative text](./app_styling/f_g.jpg)


#### Project Status: [Active] 

## Project Intro/Objective
The purpose of this project is to write an algorithm to detect fruits and vegetables.

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

## Project Description
First I used TensorFlow Object Detection API to detect multiple fruits and vegetables in one picture, this is an opensource computer vision framework for building object detection and image segmentation models that can localize multiple objects in the same image, here I used VGG16, ResNet50, and EfficientNetB0 (not the best performance).
I also trained a Convolutional Neural Network on the dataset, it performs rather well on the test data with 99.20% accuracy but not he best detection performance when I apply it on a single uploaded image. Same thing applies for the MobileNetV2 and VGG-16 transfer learning models.
VGG-16: was not included in the streamlit app because it was too big but you can find it in the notebook[a relative link](notebooks/Fruit_vegetables_detection_final.ipynb).


### Streamlit App Link:
https://ansamz-fruit-vegetables-detector-fruit-veg-app-2jocy1.streamlit.app/

## TODO
YOLO
You Only Look Once:
Unified, Real-Time Object Detection
Further reading: 
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf