from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
import cv2
import keras
import numpy as np
import os
from keras.optimizers import Adam
from directory_management import *
from models_configuration import *
# import image_processing as ip
import image_threshold as it
from PIL import ImageFont, ImageDraw, Image

apparel_model_path = "bottleneck_fc_model.h5"

apparel_labels = ["Batik", "Jacket", "Polo", "Shirt", "T-Shirt"]

img_width, img_height = 150, 150
b, g, r, a = 255, 255, 255, 0

# for feature extraction
pre_model = applications.VGG16(include_top=False, weights='imagenet')

def load_models():
	apparel = load_model(apparel_model_path)

	adam = Adam(lr=0.0001)
	apparel.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	return apparel

apparel_model = load_models()

def resize_image(event):
	new_width = event.width
	new_height = event.height
	image = copy_of_image.resize((new_width, new_height))
	photo = ImageTk.PhotoImage(image)
	label.config(image = photo)
	label.image = photo #avoid garbage collection

def loadimage(input_image, image_source):
	if image_source == 'camera':
		cv2.imwrite('temp.jpg', input_image)
		img = getfeature('temp.jpg')
	elif image_source == 'image':
		img = getfeature(input_image)
	return img

def show_top_three(class_prob):
	pred_list = np.argsort(class_prob)[0]
	print (pred_list)
	topidx = []
	toplabels = []
	j = 0
	labels = apparel_labels
	for i in range(-1, -4, -1):
		idx = pred_list[i]
		topidx.append(idx)
		toplabels.append(labels[idx])
		print(topidx[j])
		print(toplabels[j])
		j += 1
	return topidx, toplabels

def getprediction(input_image, image_source):
	img = loadimage(input_image, image_source)
	prediction, topidx, toplabels = [], [], []
	# predict result
	prediction = apparel_model.predict(img)
	topidx, toplabels = show_top_three(prediction)
	print(prediction)
	return toplabels

# extract feature from image
def getfeature(input_image):
    img = image.load_img(input_image, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = pre_model.predict(x)
    return features

def live_processing():
    cap = cv2.VideoCapture(0)
    cam_width = cap.get(3)  # float
    cam_width = int(cam_width)
    window_width = cam_width
    cam_height = cap.get(4) # float
    cam_height = int(cam_height)
    window_height = cam_height
    while True:
        ret, frame = cap.read()
        cv2.rectangle(frame, (140, 100), (500, 380), (0, 255, 0), 2)        
        croppedframe = frame[100:-100, 140:-140]
    
        appareltoplabels = getprediction(croppedframe, "camera")
        fontpath = "C:\\Users\\ACER\\Desktop\\COMP6065-Artificial_Intelligence\\ProjectAI\\image-classification\\fonts\\VAGRundschrift.ttf"
        font = ImageFont.truetype(fontpath, 28)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        # draw.text((145, 95), 'press SPACE to capture an image', font = font, fill = (b, g, r, a))
        draw.text((10, window_height - 96), '#1 ' + appareltoplabels[0], font = font, fill = (b, g, r, a))
        draw.text((10, window_height - 68), '#2 ' + appareltoplabels[1], font = font, fill = (b, g, r, a))
        draw.text((10, window_height - 40), '#3 ' + appareltoplabels[2], font = font, fill = (b, g, r, a))
        img = np.array(img_pil)

        cv2.imshow('camera', img)

        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == 32:
            ret, currframe = cap.read()
            cv2.imwrite('capture.jpg', currframe)
            cap.release()
            cv2.destroyAllWindows()
            break
    return appareltoplabels