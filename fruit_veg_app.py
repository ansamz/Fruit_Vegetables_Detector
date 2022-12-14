#import libraries

import streamlit as st

import pandas as pd
import numpy as np
import copy

from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import transform

import os
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from sklearn.metrics import accuracy_score
from PIL import Image
from skimage import transform

from sklearn.metrics import accuracy_score


#Add title and header
#Add page configuration
st.set_page_config(page_title="Fruits and Vegetables Image Detector", # page title, displayed on the window/tab bar
        		   page_icon="apple", # favicon: icon that shows on the window/tab bar (tip: you can use emojis)
                   layout="wide", # use full width of the page
                   menu_items={
                       'About': "App using various models to detect fruits and vegetables"
                   })

st.markdown("<h1 style='text-align: center; color: purple;'>Fruits and Vegetables Image Detector</h1>", unsafe_allow_html=True)

#Add sidebar to app
with st.sidebar: 
    st.image("./app_styling/f_g.jpg")
    st.subheader("Which model would you like to use?")
    model_choice = st.radio("Prediction model: ", ["CNN", "MobileNet"]) #, "VGG" (canceled due to large file)
    image_choice = st.radio("How would you like to test the models? ", ["Test Data", "Upload One Image"])
    #give choice to also run the model on test data/upload a whole folder
    predict = st.button("Show prediction")

    st.subheader("Prediction using Object Detection API")
    st.write('This is suitable for multiple objects prediction')
    model2_choice = st.radio("Object Detection model: ", ["VGG16", 'ResNet50', 'EfficientNetB0'])
    predict2 = st.button("Show Multiple Predictions")

    file = st.file_uploader("Upload an Image")

###
#path for images
###

train_dir = Path('./data/Training')
train_lst = list(train_dir.glob(r'**/*.jpg'))

test_dir = Path('./data/Test')
test_lst = list(test_dir.glob(r'**/*.jpg'))

###
#Functions
###

def df_images(lst_filepaths):
    """ 
    from the list above create a dataframe containing a coulmn of the path and labels of the pictures
    """

    labels = [str(lst_filepaths[i]).split("/")[-2] for i in range(len(lst_filepaths))]

    lst_filepaths = pd.Series(lst_filepaths, name='path').astype(str)
    labels = pd.Series(labels, name='label')

    df = pd.concat([lst_filepaths, labels], axis=1)

    # Shuffle dataframe in-place and reset index 
    df = df.sample(frac=1).reset_index(drop = True)
    
    return df


def display_picture(df):
  df2 = df.copy(deep=True)
  df2 = df2.drop_duplicates(subset=["label"]).reset_index()

  fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 10),
                        subplot_kw={'xticks': [], 'yticks': []})
  
  for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df2.path[i]))
    ax.set_title(df2.label[i], fontsize = 15)
  plt.tight_layout()
  #plt.show()

  return fig

def acc_score_display(model):
  # Predict the label of the test_images
  pred = model.predict(test_images)
  pred = np.argmax(pred,axis=1)

  # Map the label
  labels = (train_images.class_indices)
  labels = dict((v,k) for k,v in labels.items())
  pred = [labels[k] for k in pred]

  y_test = [labels[k] for k in test_images.classes]

  acc = accuracy_score(y_test, pred)
  acc_result = f'Accuracy on the test set: {100*acc:.2f}%'

  # Display some pictures of the dataset with their labels and the predictions
  fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15),
                          subplot_kw={'xticks': [], 'yticks': []})

  for i, ax in enumerate(axes.flat):
      ax.imshow(plt.imread(test_df.path.iloc[i]))
      ax.set_title(f"True: {test_df.label.iloc[i]}\nPredicted: {pred[i]}")
  plt.tight_layout()
  #plt.show()

  return fig, acc_result
  
def load(path):
   np_image = Image.open(path)
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, 0)
   return np_image
  
def single_image_pred(model, path):
  image = load(path)
  predictions = model.predict(image)
  pred = np.argmax(predictions, axis=1)
  labels = (train_images.class_indices)
  class_names = dict((v,k) for k,v in labels.items())
  result = "This image is most likely a {} with a {:.2f} percent confidence.".format(class_names[np.argmax(predictions, axis=1)], 100 * np.max(predictions))
  return result, image

def get_predictions(model, img_path):
    f, ax = plt.subplots()
    f.set_size_inches(12, 8)
    ax.imshow(Image.open(img_path).resize((50, 50), Image.ANTIALIAS))
    
    f2, axes = plt.subplots()
    f.set_size_inches(12, 8)
    img = load(img_path)
    img = preprocess_input(img)
    preds  = decode_predictions(model.predict(img), top=3)[0]
    b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes)
    b.tick_params(labelsize=55)
    return f, f2


###
#generate batches from dataframes
###

train_df = df_images(train_lst)
test_df = df_images(test_lst)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42, #42 the answer to everything :D 
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

###
#Load models
###
@st.cache(allow_output_mutation=True)
def load_model(model):
    return keras.models.load_model(model)


#Model is too big for github so I cancelled it
#vgg16 = keras.models.load_model("vggmodel")

###
#predict
###

agree = st.checkbox("I would like to see examples of the training data used")

if agree:
    fig1 = display_picture(train_df)
    st.pyplot(fig1)

if predict:
    cnn_model = load_model("cnn_model")
    mobilenet_model = load_model("mobilenet_model")
    if model_choice == "CNN":
        if image_choice == "Upload One Image":
            with st.spinner('PREDICTING! I have grape expectations for this.'):
                result, image = single_image_pred(cnn_model, file)
                st.image(file, caption='The image you selected')
                st.header(result)
        elif image_choice == "Test Data":
            with st.spinner('PREDICTING! still early to celery-brate'):
                fig2, acc_result = acc_score_display(cnn_model)
                st.header(acc_result)
                st.pyplot(fig2)
    elif model_choice == "MobileNet":
        if image_choice == "Upload One Image":
            with st.spinner('PREDICTING! Peas don’t go.'):
                result, image = single_image_pred(mobilenet_model, file)
                st.image(file, caption='The image you selected')
                st.header(result)
        elif image_choice == "Test Data":
            with st.spinner('PREDICTING! Lettuce pray'):
                fig2, acc_result = acc_score_display(mobilenet_model)
                st.header(acc_result)
                st.pyplot(fig2)
    # elif model_choice == "VGG":
    #     if image_choice == "Upload One Image":
    #         with st.spinner('PREDICTING!I ap-peach-iate your patience'):
    #             result, image = single_image_pred(vgg16, file)
    #             st.image(file, caption='The image you selected')
    #             st.header(result)
    #     elif image_choice == "Test Data":
    #         with st.spinner('PREDICTING! Keep calm and carrot on'):
    #             fig2, acc_result = acc_score_display(vgg16)
    #             st.header(acc_result)
    #             st.pyplot(fig2)

if predict2:
    if model2_choice == "VGG16":
        with st.spinner('PREDICTING! I ap-peach-iate your patience'):
            vgg16_model = VGG16(weights="imagenet")
            f, f2 = get_predictions(vgg16_model, file)
            st.pyplot(f)
            st.pyplot(f2)
            
            
    elif model2_choice == "ResNet50":
        with st.spinner('PREDICTING! I ap-peach-iate your patience'):
            resnet_model = ResNet50(weights="imagenet")
            f, f2 = get_predictions(resnet_model, file)
            st.pyplot(f)
            st.pyplot(f2)

    elif model2_choice == 'EfficientNetB0':
        with st.spinner('PREDICTING! I ap-peach-iate your patience'):
            efnetb0_model = EfficientNetB0(weights="imagenet")
            f, f2 = get_predictions(efnetb0_model, file)
            st.pyplot(f)
            st.pyplot(f2)
