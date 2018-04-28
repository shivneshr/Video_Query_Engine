

# Conputer vision and utility Libraries
import cv2
import numpy as np
import os
import pickle

# Deeplearning Libraries and models

from keras.applications import vgg16
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


# Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')

# Layer name from which the vectors needs to be extracted
layer_name = 'fc2'

# Specifying the output variable
intermediate_layer_model = Model(inputs=vgg_model.input,
								 outputs=vgg_model.get_layer(layer_name).output)

# Load the Inception_V3 model
# inception_model = inception_v3.InceptionV3(weights='imagenet')

# Load the ResNet50 model
# resnet_model = resnet50.ResNet50(weights='imagenet')

# Load the MobileNet model
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')



def extract_CNN_features(path1,path2):

	original = load_img(path1, target_size=(224, 224))
	original_2 = load_img(path2, target_size=(224, 224))
	numpy_image = img_to_array(original)
	numpy_image_2 = img_to_array(original_2)

	image_batch = np.expand_dims(numpy_image, axis=0)
	image_batch_2 = np.expand_dims(numpy_image_2, axis=0)
	# prepare the image for the VGG model
	processed_image = vgg16.preprocess_input(image_batch.copy())
	processed_image_2 = vgg16.preprocess_input(image_batch_2.copy())

	intermediate_output = intermediate_layer_model.predict(processed_image)
	print(intermediate_output)
	intermediate_output_2 = intermediate_layer_model.predict(processed_image_2)
	print(intermediate_output_2)

	diff = np.absolute(intermediate_output - intermediate_output_2)
	sum = np.sum(diff)

	print(sum)


extract_CNN_features("keyframes/musicvideo/musicvideo_frame6.jpg","keyframes/first/first_frame3.jpg")