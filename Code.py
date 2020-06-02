# -- coding: utf-8 --
"""
Created on Wed Mar 18 20:02:19 2020

@author: Developer
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "C:/Users/Developer/Documents/Internships/Gaia/Resize/Original Samples"

CATEGORIES = ["Black", "Blue", "Cracked", "Pale", "White"]

training_data = []

IMG_SIZE=64

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=black 1=blue

        for img in os.listdir(path):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_training_data()

print(len(training_data))

import random
random.shuffle(training_data)
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)




import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D




X = X/255.0  #normalising the data by scaling

model = Sequential()   #sequential model

#2 times layering

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors



model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam')

model.fit(X, y, batch_size=13, epochs=4, validation_split=0.1) #90 train 10 test




loss = abs( model.evaluate(X, y, verbose = 0) )
print('Test loss:', loss)
print('Test accuracy:', 100-loss)