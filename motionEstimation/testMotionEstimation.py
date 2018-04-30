import cv2 as cv
import numpy as np

frames = ['Q4.MP4_frame1.jpg', 'Q4.MP4_frame2.jpg', 'Q4.MP4_frame3.jpg', 'Q4.MP4_frame4.jpg', 'Q4.MP4_frame5.jpg',
		  'Q4.MP4_frame6.jpg', 'Q4.MP4_frame7.jpg', 'Q4.MP4_frame8.jpg', 'Q4.MP4_frame9.jpg']

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
					  qualityLevel=0.3,
					  minDistance=7,
					  blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
				 maxLevel=2,
				 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
old_gray = cv.imread('../keyframes_query/Q4.MP4/' + frames[0], 0)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_gray)

for frameName in frames[1:]:
	frame = cv.imread('../keyframes_query/Q4.MP4/' + frameName, 0)

	frame_gray = frame

	# calculate optical flow
	p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	# Select good points

	change_in_x = p1[:, :, 0]
	change_in_y = p1[:, :, 1]

	good_new = p1[st == 1]
	good_old = p0[st == 1]

	# draw the tracks
	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a, b = new.ravel()
		c, d = old.ravel()
		mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
		frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
	img = cv.add(frame, mask)
	cv.imshow('frame', img)
	k = cv.waitKey(30) & 0xff
	if k == 27:
		break
	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1, 1, 2)
cv.destroyAllWindows()
