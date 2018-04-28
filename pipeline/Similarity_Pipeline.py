

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

currentFolder = ""
previousFolder = ""

featureList = []

# only if you want to keyframe extraction for specific folders
onlyfor = ['first','second']



p_frame_thresh = 98000  # You may need to adjust this threshold for keyframes extraction (lower -> more frames)


def extractKeyFrames(video_path,videoName):
	global p_frame_thresh

	cap = cv2.VideoCapture(video_path,0)
	# Read the first frame.
	ret, prev_frame = cap.read()


	count = 0
	while ret:
		prev_gray_image = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
		ret, curr_frame = cap.read()

		if ret:
			curr_gray_image = cv2.cvtColor(curr_frame,cv2.COLOR_RGB2GRAY)
			diff = cv2.absdiff(curr_gray_image, prev_gray_image)
			non_zero_count = np.count_nonzero(diff)
			if non_zero_count > p_frame_thresh:
				count += 1
				if not os.path.exists("keyframes/"+videoName):
					os.makedirs("keyframes/"+videoName)
				cv2.imwrite("keyframes/%s/%s_frame%d.jpg" % (videoName,videoName,count), curr_frame)
				print(count, "Got P-Frame")
			prev_frame = curr_frame
			prev_gray_image = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)


def iterateThroughDirectory(directory):

	for files in os.listdir(directory):
		print("Processing ",files)
		if os.path.isfile(directory + files):
			extractKeyFrames(directory+files,files.replace(".mp4",""))


def write_to_pickle_file(folder):
	global featureList, currentFolder, previousFolder

	location = 'keyframes/'+folder+'/'+folder+'.pickle'
	with open(location, 'wb') as handle:
		pickle.dump(featureList, handle, protocol=pickle.HIGHEST_PROTOCOL)
	currentFolder = previousFolder
	featureList = []


def findDifference(queryPath,databasePath):

	queryFeatures = []
	databaseFeatures = []
	queryFileName = queryPath.split('/')[-1]
	databaseFileName = databasePath.split('/')[-1]

	with open(queryPath+'/'+queryFileName+'.pickle', 'rb') as handle:
		queryFeatures = pickle.load(handle)

	with open(databasePath+'/'+databaseFileName+'.pickle', 'rb') as handle:
		databaseFeatures = pickle.load(handle)

	minDist = float('inf')

	for feature_d in databaseFeatures:
		for feature_q in queryFeatures:
			diff = np.absolute(feature_d - feature_q)
			sum = np.sum(diff)
			print(sum)
			if sum < minDist:
				minDist = sum


	print("match factor ",minDist)


def extract_CNN_features(path):

	global  previousFolder, currentFolder, featureList

	for root, dirs, files in os.walk(path):

		if previousFolder != "" and currentFolder != previousFolder:
			write_to_pickle_file(previousFolder)

		for name in files:

			if not (len(onlyfor)==0 or (len(onlyfor)!=0 and (root.split('/')[-1] in onlyfor))):
				break

			print(os.path.join(root, name))


			if name.endswith(".jpg"):
				previousFolder = root.split('/')[-1]
				# Load an image in PIL format
				original = load_img(os.path.join(root, name), target_size=(224, 224))

				# convert the PIL image to a numpy array
				# IN PIL - image is in (width, height, channel)
				# In Numpy - image is in (height, width, channel)

				numpy_image = img_to_array(original)

				# Convert the image / images into batch format
				# expand_dims will add an extra dimension to the data at a particular axis
				# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
				# Thus we add the extra dimension to the axis 0.

				image_batch = np.expand_dims(numpy_image, axis=0)

				# prepare the image for the VGG model
				processed_image = vgg16.preprocess_input(image_batch.copy())

				intermediate_output = intermediate_layer_model.predict(processed_image)
				featureList.append(intermediate_output[0])

		if previousFolder != "" and currentFolder != previousFolder:
			write_to_pickle_file(previousFolder)



# Extracts the keyframes of the set of videos and stores result in keyframes folder
#iterateThroughDirectory("video/")

# Extracts the keyframes of the set of videos and stores result in keyframes folder
#iterateThroughDirectory("query/")

# Extracts the CNN features for all the video files and stores in Pickle file in corresponding directory
#extract_CNN_features("keyframes/")

# Finds the similarity between any two given folders (lower the score better similarity)
findDifference("keyframes/second","keyframes/musicvideo")







