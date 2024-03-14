import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 
import numpy as np
import streamlit as st
import os
from tensorflow.keras.utils import load_img
from keras.models import load_model
from keras.preprocessing import image
import pickle

def app():
    
    # Define image size and batch size
    IMG_SIZE = 224
    BATCH_SIZE = 32

    # Define data generators for train, validation and test sets
    train_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_datagen = ImageDataGenerator(rescale = 1./255)

    val_generator = val_datagen.flow_from_directory(
        'val',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    {
    'name': 'expanded_conv_depthwise',
    'trainable': True,
    'dtype': 'float32',
    'kernel_size': [3, 3],
    'strides': [1, 1],
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': [1, 1],
    'activation': 'linear',
    'use_bias': False,
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'bias_regularizer': None,
    'activity_regularizer': None,
    'bias_constraint': None,
    'depth_multiplier': 1,
    'depthwise_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None}},
    'depthwise_regularizer': None,
    'depthwise_constraint': None
    }

    # Define the model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
         train_generator,
         validation_data=val_generator,
         epochs=10
     )
    model.save("Model1.h5","label.txt")
    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(test_generator)
    #print('Test accuracy:', test_acc)
    np.save('Model1_acc',test_acc)

'''
    # predicting images
    img = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    st.info(classes)
    #print classes

    # predicting multiple images at once
    img = image.load_img('test2.jpg', target_size=(img_width, img_height))
    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)

    # pass the list of multiple images np.vstack()
    images = np.vstack([x, y])
    classes = model.predict_classes(images, batch_size=10)'''
    

app()
