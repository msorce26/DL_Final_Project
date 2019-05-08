# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:51:25 2019

@author: msorc
"""

import numpy as np
from random import shuffle
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy.ndimage import zoom



filename = 'cropped_data0.csv'

#Converts a JPG image to an array
def jpg_image_to_array(image_path):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(image_path) as image:         
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
  return im_arr


#######################################################################
#Imports the CSV and separates the feature data from the labels
import csv
 
xdata = []
ydata = []
 
with open(filename) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        xdata.append(row[0])
        ydata.append(row[1])

del xdata[0]
del ydata[0]

images = []
for index in range(len(xdata)):
    rgb = jpg_image_to_array(xdata[index])
#    grey = rgb2gray(rgb) #Uncomment and change variable for grayscale
#    grey = grey.reshape(25, 25, 1)
    images.append(rgb)
images = np.asarray(images)
ydata = np.asarray(ydata)

#for index in range(len(images)):
images = zoom(images,(1,5,7,1))


x_train = images[0:9560]
x_test = images[9561:11943]
y_train = np.array(ydata[0:9560])
y_test = np.array(ydata[9561:11943])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#test = [int(string_num) for string_num in y_train]

#x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (35, 35, 3)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

######################################################################
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(3, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(3, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(128, activation=tf.nn.relu))
#model.add(Dropout(0.2))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(10,activation=tf.nn.softmax))


############################################################################
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
history = model.fit(x=x_train,y=y_train, epochs=10, validation_data = (x_test,y_test))



##########################################################################
#model.evaluate(x_test, y_test)
print('Test Module Evaluate', model.evaluate(x_test,y_test))
#model.summary()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
