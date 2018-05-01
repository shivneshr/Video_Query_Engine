# Conputer vision and utility Libraries
import cv2
import numpy as np
import os
import sys
import pickle
import re
import platform

# Deeplearning Libraries and models
from keras.applications import vgg16
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.imagenet_utils import decode_predictions

fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"

# User defined functions

UI_CNN_ScriptPath = ".."+fileSeparator+"CNNRelation"+fileSeparator
UI_RGB_ScriptPath = ".."+fileSeparator+"RGBRelation"+fileSeparator
UI_AUD_ScriptPath = ".."+fileSeparator+"soundAnalysis"+fileSeparator

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(UI_CNN_ScriptPath))
sys.path.append(os.path.abspath(UI_RGB_ScriptPath))
sys.path.append(os.path.abspath(UI_AUD_ScriptPath))

import GenerateCNN_Plots as cnn
import rgbRelationBruteForce as rgb
import soundRelationUsingDatabase as audio


# Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')

# Layer name from which the vectors needs to be extracted
layer_name = 'fc2'

# Specifying the output variable
intermediate_layer_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(layer_name).output)

# Load the Inception_V3 model
# inception_model = inception_v3.InceptionV3(weights='imagenet')

# Load the ResNet50 model
# resnet_model = resnet50.ResNet50(weights='imagenet')

# Load the MobileNet model
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')

currentFolder = ""
previousFolder = ""

featureList = []
labelList = []

# only if you want to CNN extraction for specific folders
onlyfor = []

keyframePath_database = ".."+fileSeparator+"keyframes_database"+fileSeparator
keyframePath_query = ".."+fileSeparator+"keyframes_query"+fileSeparator
queryPath = ".."+fileSeparator+"videos"+fileSeparator+"query"+fileSeparator
databasePath = ".."+fileSeparator+"videos"+fileSeparator+"search"+fileSeparator

p_frame_thresh = 90000  # You may need to adjust this threshold for keyframes extraction (lower -> more frames)

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def extractKeyFrames(video_path, videoName, keyframePath):
    global p_frame_thresh

    cap = cv2.VideoCapture(video_path, 0)
    # Read the first frame.
    ret, prev_frame = cap.read()

    if not os.path.exists(keyframePath + videoName):
        os.makedirs(keyframePath + videoName)

    count = 0
    while ret:
        prev_gray_image = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        ret, curr_frame = cap.read()

        if ret:
            curr_gray_image = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(curr_gray_image, prev_gray_image)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count > p_frame_thresh:
                count += 1
                cv2.imwrite(keyframePath + "%s"+fileSeparator+"%s_frame%d.jpg" % (videoName, videoName, count), curr_frame)
                print(count, "Got P-Frame")
            prev_frame = curr_frame
            prev_gray_image = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(keyframePath + "%s"+fileSeparator+"%s_frame%d.jpg" % (videoName, videoName, count), prev_frame)


def extractKeyFrames_Second(video_path, videoName, keyframePath):
    cap = cv2.VideoCapture(video_path)
    # Read the first frame.
    ret, frame = cap.read()

    multiplier = 15

    if not os.path.exists(keyframePath + videoName):
        os.makedirs(keyframePath + videoName)

    count = 0
    while ret:
        frameId = int(round(cap.get(1)))
        ret, frame = cap.read()

        if frameId % multiplier == 0 and ret:
            count += 1
            cv2.imwrite(keyframePath + "%s"+fileSeparator+"%s_frame%d.jpg" % (videoName, videoName, count), frame)

    cap.release()
    print("Complete")


def iterateThroughDirectory(directory, keyframepath):
    for files in os.listdir(directory):
        print("Processing ", files)
        if os.path.isfile(directory + files):
            extractKeyFrames_Second(directory + files, files.replace(".mp4", ""), keyframepath)


def write_to_pickle_file(folder, keyframePath):
    global featureList, labelList, currentFolder, previousFolder

    location = keyframePath + folder + fileSeparator + folder + '.pickle'
    lableLocation = keyframePath + folder + fileSeparator + folder + '_label.pickle'

    with open(location, 'wb') as handle:
        pickle.dump(featureList, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(lableLocation, 'wb') as handle:
        pickle.dump(labelList, handle, protocol=pickle.HIGHEST_PROTOCOL)

    currentFolder = previousFolder
    featureList = []
    labelList = []


def find_match_CNN(queryName):
    queryPath = keyframePath_query + queryName
    databasePath = keyframePath_database
    queryFeatures = []
    databaseFeatures = []
    queryFileName = queryPath.split(fileSeparator)[-1]
    databaseFileName = databasePath.split(fileSeparator)[-1]

    with open(queryPath + fileSeparator + queryFileName + '.pickle', 'rb') as handle:
        queryFeatures = pickle.load(handle)

    match_factor = {}

    for folder in os.listdir(databasePath):

        with open(databasePath + folder + fileSeparator + folder + '.pickle', 'rb') as handle:
            databaseFeatures = pickle.load(handle)

        minDist = float('inf')
        min_index_d = -1
        min_index_q = -1

        for index_d, feature_d in enumerate(databaseFeatures):
            for index_q, feature_q in enumerate(queryFeatures):
                diff = np.absolute(feature_d - feature_q)
                sum = np.sum(diff)
                if sum < minDist:
                    minDist, min_index_d, min_index_q = sum, index_d + 1, index_q + 1

        segment = min_index_d - min_index_q
        if segment < 1:
            segment = 1
        match_factor[folder] = {"distance": minDist, "index_d": min_index_d, "index_q": min_index_q, "segment": segment}


    rankedOutput = []
    for key, value in sorted(match_factor.items(), key=lambda x: x[1]["distance"])[:3]:
        rankedOutput.append((key,[(value['segment'] / 2.0) - 0.5, ((value['segment'] + 8 )/ 2.0) + 0.5]))


    cnn.generateVideoPlots(rankedOutput, queryName, 'CNN')
    rgb.generateVideoHistogram(rankedOutput, queryName, 'CNN')
    audio.generateAudioMfccImages(rankedOutput, queryName, 'CNN')


# print("%s: %s" % (key, value))


def extract_CNN_features(path):
    global previousFolder, currentFolder, featureList

    for root, dirs, files in os.walk(path):

        if previousFolder != "" and currentFolder != previousFolder:
            write_to_pickle_file(previousFolder, path)

        files = sorted([file for file in files if file.endswith('.jpg')], key=numericalSort)
        for name in files:

            if not (len(onlyfor) == 0 or (len(onlyfor) != 0 and (root.split(fileSeparator)[-1] in onlyfor))):
                break

            print(os.path.join(root, name))

            previousFolder = root.split(fileSeparator)[-1]
            # Load an image in PIL format
            original = load_img(os.path.join(root, name), target_size=(224, 224))

            # convrt the PIL image to a numpy array
            # IN PL - image is in (width, height, channel)
            # In Nmpy - image is in (height, width, channel)

            numpy_image = img_to_array(original)

            # Convert the image \\ images into batch format
            # expand_dims will add an extra dimension to the data at a particular axis
            # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
            # Thus we add the extra dimension to the axis 0.

            image_batch = np.expand_dims(numpy_image, axis=0)

            # prepare the image for the VGG model

            processed_image = vgg16.preprocess_input(image_batch.copy())
            intermediate_output = intermediate_layer_model.predict(processed_image)
            predictions = vgg_model.predict(processed_image)

            # print predictions
            # convert the probabilities to class labels
            # We will get top 5 predictions which is the default

            label = decode_predictions(predictions)
            labelList.append(label[0])
            featureList.append(intermediate_output[0])

        if previousFolder != "" and currentFolder != previousFolder:
            write_to_pickle_file(previousFolder, path)


# Extracts the keyframes of the set of videos and stores result in keyframes folder
# iterateThroughDirectory(databasePath,keyframePath_database)

# Extracts the keyframes of the set of videos and stores result in keyframes folder
# iterateThroughDirectory(queryPath,keyframePath_query)

# Extracts the CNN features for all the video files and stores in Pickle file in corresponding directory

# For database videos
# extract_CNN_features(keyframePath_database)

# For Query videos

# extract_CNN_features(keyframePath_query)

# Finds the similarity between any given query video & database folders (lower the score better similarity)

#find_match_CNN("Q4.MP4")
