import cv2
import numpy as np
import os

p_frame_thresh = 98000  # You may need to adjust this threshold


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


iterateThroughDirectory("videos/")


