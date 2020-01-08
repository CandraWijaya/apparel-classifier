import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import applications
from keras.utils.np_utils import to_categorical

# import models_configuration as mc
import tensorflow as tf
import math
import os
import time
import cv2
import plot as p

start = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = r'data/train'
validation_data_dir = r'data/validation'
nb_train_samples = 5000
nb_validation_samples = 2000
epochs = 50
batch_size = 32
classes = 5

tf.device('/device:GPU:0')

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(classes, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model.add(top_model)
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save("fc_model.h5")

end = time.time()

print('Training time : ', (end - start) // 60, ' minutes')

# summarize history for accuraccy
p.gettraingraph(model, 'acc', 1, 'fine-tuning')

# summarize history for loss
p.gettraingraph(model, 'loss', 1, 'fine-tuning')

model.summary()