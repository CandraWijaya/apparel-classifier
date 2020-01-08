import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
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
train_feature = r'bottleneck_features_train.npy'
validation_feature = r'bottleneck_features_validation.npy'
nb_train_samples = 5000
nb_validation_samples = 2000
epochs = 50
batch_size = 32
classes = 5

tf.device('/device:GPU:0')

from keras import backend as K
K.set_image_dim_ordering('th')

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_generator = datagen.flow_from_directory(
        train_data_dir, 
        target_size=(img_width, img_height), 
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle=False)
    predict_size_train = int(math.ceil(train_generator.samples / batch_size))
    # predict_size_train = train_generator.samples // batch_size
    bottleneck_features_train = model.predict_generator(train_generator, predict_size_train, verbose = 1)
    # np.save(open(train_feature, 'wb'), bottleneck_features_train)
    np.save(train_feature, bottleneck_features_train)

    val_generator = datagen.flow_from_directory(
        validation_data_dir, 
        target_size=(img_width, img_height), 
        batch_size=batch_size,
        class_mode='categorical', 
        shuffle=False)
    predict_size_validation = int(math.ceil(val_generator.samples / batch_size))
    # predict_size_validation = val_generator.samples // batch_size
    bottleneck_features_validation = model.predict_generator(val_generator, predict_size_validation, verbose = 1)
    np.save(validation_feature, bottleneck_features_validation)
    # np.save(open(validation_feature, 'wb'), bottleneck_features_validation)

def train_top_model():
    train_data = np.load(train_feature)
    train_labels = np.array([0] * 1000 + [1] * 1000 + [2] * 1000 + [3] * 1000 + [4] * 1000)

    validation_data = np.load(validation_feature)
    validation_labels = np.array([0] * 400 + [1] * 400 + [2] * 400 + [3] * 400 + [4] * 400)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name='prediction'))

    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    train_labels = to_categorical(train_labels, classes)
    validation_labels = to_categorical(validation_labels, classes)

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose = 1)
    model.save(top_model_weights_path)
    return model

save_bottlebeck_features()
model = train_top_model()

end = time.time()

print('Training time : ', (end - start) // 60, ' minutes')

# summarize history for accuraccy
p.gettraingraph(model, 'acc', 1, 'feature-extraction')

# summarize history for loss
p.gettraingraph(model, 'loss', 1, 'feature-extraction')