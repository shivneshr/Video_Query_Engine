import glob
import cv2
import matplotlib.pyplot as plt
import platform
from subprocess import Popen

fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"

def findHist(imagePath):
    image = cv2.imread(imagePath)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def getMatchingFrames(queryPath, diffMethod):

    queryHist = {}
    index = 1
    for queryKeyFrameNames in glob.glob(queryPath + "*.jpg"):
        queryHist[index] = findHist(queryKeyFrameNames)
        index = index+1

    videoSums = {}
    videoKeyFrame = {}

    for dir in glob.glob(".."+fileSeparator +"keyframes_database"+fileSeparator+"*"+fileSeparator):
        videoHist = {}
        reverse = False
        for i in range(1,40):
            videoKeyFrameNames = dir + dir.split(fileSeparator)[-2] + "_frame" + str(i) + ".jpg"
            videoHist[i] = findHist(videoKeyFrameNames)

        if diffMethod in [cv2.HISTCMP_CORREL,cv2.HISTCMP_INTERSECT]:
            reverse = True

        videoMaxSum = 0
        for i in range(0,31):
            sum = 0
            for j in range(1, 10):
                sum += cv2.compareHist(queryHist[j], videoHist[i+j], diffMethod)
            if(sum > videoMaxSum):
                videoMaxSum = sum
                videoBestFrame = [i+1,i+9]

        videoSums[dir.split(fileSeparator)[-2]] = videoMaxSum
        videoKeyFrame[dir.split(fileSeparator)[-2]] = videoBestFrame

    sortedSums = [(k, videoSums[k]) for k in sorted(videoSums, key=videoSums.get, reverse=reverse)]

    print("Best match video : " + sortedSums[0][0])
    print("Best match keyframes : " + str(videoKeyFrame[sortedSums[0][0]]))

    return sortedSums[0][0], videoKeyFrame[sortedSums[0][0]]


# Please give directory name of the query
queryDirName = "first"
queryPath = ".." + fileSeparator + "keyframes_query"+fileSeparator+ queryDirName + fileSeparator
videoDirName, bestMatchFrames = getMatchingFrames(queryPath, cv2.HISTCMP_CORREL)

for i in range(bestMatchFrames[0], bestMatchFrames[1]+1):
    filename = ".."+fileSeparator + "keyframes_database" + fileSeparator + videoDirName + fileSeparator + videoDirName + "_frame" + str(i) +".jpg"
    img = cv2.imread(filename)
    color = ('b', 'g', 'r')
    for j, col in enumerate(color):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.savefig("video-00" + str(i + 1 - bestMatchFrames[0]) + ".png")
    plt.close()

index = 1
for filename in glob.glob(queryPath + "*.jpg"):
    img = cv2.imread(filename)
    color = ('b', 'g', 'r')
    for j, col in enumerate(color):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.savefig("query-00" + str(index) + ".png")
    index = index+1
    plt.close()

if platform.system() == "Windows":
    p = Popen("convertToVideo.bat")
else:
    p = Popen("convertToVideo.sh")
p.communicate()