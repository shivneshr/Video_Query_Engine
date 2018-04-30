import os
import re
import pickle

import cv2 as cv
import numpy as np

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
					  qualityLevel=0.3,
					  minDistance=7,
					  blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
				 maxLevel=2,
				 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

numbers = re.compile(r'(\d+)')


def numericalSort(value):
	parts = numbers.split(value)
	parts[1::2] = map(int, parts[1::2])
	return parts


def calculate_motion_vectors(path):
	global feature_params, lk_params

	filelist = sorted([file for file in os.listdir(path) if file.endswith('.jpg')], key=numericalSort)

	motionvector = []

	for index in range(len(filelist) - 1):
		frame1 = cv.imread(path +'/'+ filelist[index], 0)
		frame2 = cv.imread(path +'/'+ filelist[index + 1], 0)

		# Getting the features in frame 1 to track
		p0 = cv.goodFeaturesToTrack(frame1, mask=None, **feature_params)

		# calculate optical flow
		p1, st, err = cv.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)

		motionvector.append(np.sum(np.absolute(p1)))

	location = path + '/' + path.split('/')[-1] + '_motion.pickle'
	with open(location, 'wb') as handle:
		pickle.dump(motionvector, handle, protocol=pickle.HIGHEST_PROTOCOL)


def motion_estimation(parent_path):

	for dirs in os.listdir(parent_path):
		calculate_motion_vectors(parent_path+'/'+dirs)


def motion_vector_matching(queryPath,databasePath):

	queryFeatures = None
	databaseFeatures = None
	queryFileName = queryPath.split('/')[-1]


	with open(queryPath+'/'+queryFileName+'_motion.pickle', 'rb') as handle:
		queryFeatures = pickle.load(handle)


	querysum = sum(queryFeatures)
	window = len(queryFeatures)
	match_factor = {}


	for folder in os.listdir(databasePath):

		location = databasePath + '/' + folder +'/'+ folder +'_motion.pickle'
		with open(location, 'rb') as handle:
			databaseFeatures = pickle.load(handle)

		minDist = float('inf')

		for index in range(len(databaseFeatures)-window):

			#print(databaseFeatures[index:index+window])
			temp = abs(sum(databaseFeatures[index:index+window]) - querysum)

			if(minDist>temp):
				minDist = temp

		match_factor[folder] = minDist



	for key, value in sorted(match_factor.items(), key=lambda x: x[1]):
		print("%s: %s" % (key, value))

#motion_estimation('../keyframes_database')
#motion_estimation('../keyframes_query')


motion_vector_matching('../keyframes_query/second','../keyframes_database')



