# This program written for Self-Driving Car ND third project 
# Behavioural Cloning. It generates a Keras mode (DAVE-2) and 
# weights file (model.h5) which are used for testing behavioral cloning
# for driving a car around tracks. The model takes input frames
# (160x320x3) and labels which is steering angles for each frame.
# Traing files are based on driving on both track in both directions.
# The model is trained to  predict the steering angle when driving around 
# track.

################################################################
# Start by importing the required libraries
################################################################

import csv
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

################################################################
# Define function here
################################################################

# Instead of storing the preprocessed data in memory all at once, 
# defined the generator to pull pieces of the data and process them 
# on the fly only when they are needed them, which is much more
# memory-efficient. 

def generator(samples, batch_size=32):
    correction_factor = 0.2
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	# Reading all three images, center, right and left
                for i in range(3):
                    name = batch_sample[i]
                    image = cv2.imread(name)
                    change_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(change_color)
                    angle = float(batch_sample[3])
                    if i == 1 :
                        angle += correction_factor
                    if i == 2 :
                        angle -= correction_factor
                    angles.append(angle)
                    
                    # augment data by flipping image
                    images.append(cv2.flip(change_color,1))
                    angles.append(angle*-1.0)
    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

################################################################
# Read Tringing data from local drive 
################################################################

lines = []
with open("../driving_data_track1/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
with open("../driving_data_track1_rev/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
with open("../driving_track2_prac2/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
with open("../deiving_track2_rev/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

################################################################
# Split data for training and validation
################################################################

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
batch_size =32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

################################################################
# Defined CNN Model  with DAVE-2 structure 
################################################################

### Use Nvidia model called DAVE-2 System from "End to End Learning for Self-Driving Cars"

model = Sequential()

# Preprocess incoming data, centered around zero with small standard 
# deviation 

model.add(Lambda(lambda x: x / 127.5 - 1.0 ,input_shape=(160,320,3)))
    
# Crop top of image since it is not useful
model.add(Cropping2D(cropping=((70,25), (0,0))))

# CNN Layer 1

model.add(Conv2D(filters=24, kernel_size=(5, 5),strides=(2,2), activation='relu'))

# CNN Layer 2
    
model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2,2), activation='relu'))
          
# CNN Layer 3
          
model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='relu'))
          
# CNN Layer 4        

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
          
# CNN Layer 5     

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
          
# Flatten

model.add(Flatten())

# FCNN Layer 1

model.add(Dense(100))
          
# FCNN Layer 2    

model.add(Dense(50))
          
# FCNN Layer 3

model.add(Dense(1))

# Compile with mse and adam

model.compile(loss='mse' , optimizer ='adam')

# Fit the model

history_object = model.fit_generator(train_generator, 
                    steps_per_epoch= len(train_samples)/batch_size,
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)/batch_size, 
                    epochs=5, verbose = 1)

# save the model
print("Saving Model as model.h5")

model.save("model.h5")



################################################################
# Plot training and validation loss 
################################################################

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()