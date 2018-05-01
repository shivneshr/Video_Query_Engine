import glob
import cv2
import matplotlib.pyplot as plt
import platform
import subprocess

fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"

def __findHist(imagePath):
    image = cv2.imread(imagePath)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def __getMatchingFrames(queryPath, diffMethod):

    queryHist = {}
    index = 1
    for queryKeyFrameNames in glob.glob(queryPath + "*.jpg"):
        queryHist[index] = __findHist(queryKeyFrameNames)
        index = index+1

    videoSums = {}
    videoKeyFrame = {}

    for dir in glob.glob(".."+fileSeparator +"keyframes_database"+fileSeparator+"*"+fileSeparator):
        videoHist = {}
        reverse = False
        for i in range(1,40):
            videoKeyFrameNames = dir + dir.split(fileSeparator)[-2] + "_frame" + str(i) + ".jpg"
            videoHist[i] = __findHist(videoKeyFrameNames)

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

    best = {}
    print(sortedSums)
    for i in range(0,3):
        tempKeyFrame = videoKeyFrame[sortedSums[i][0]]
        best[i+1] = {}
        best[i+1]["name"] = sortedSums[i][0]
        best[i+1]["timeFrames"] = [(tempKeyFrame[0]/2.0) - 0.5, (tempKeyFrame[1]/2.0) + 0.5]
        best[i+1]["keyFrames"] = tempKeyFrame
        best[i+1]["matchPercentage"] = str(sortedSums[i][1]/0.09) + "%"
    return best

# input
# queryVideoDirName Eg - "first"
def getTop3MatchedVideosWithTimeFrame(queryVideoDirName):
    queryPath = ".." + fileSeparator + "keyframes_query"+fileSeparator+ queryVideoDirName + fileSeparator
    return __getMatchingFrames(queryPath, cv2.HISTCMP_CORREL)

# input
# videoDirNameArray Eg - [ ("musicVideo",[1,6]) , ("flowers",[1.5,6.5]), ("sports",[9.5,14.5]) ]
# queryVideoDirName Eg - "first"
def generateVideoHistogram(videoDirNameTimeFrameArray, queryVideoDirName):
    queryPath = ".." + fileSeparator + "keyframes_query" + fileSeparator + queryVideoDirName + fileSeparator
    for index, videoDirNameTimeFrame in enumerate(videoDirNameTimeFrameArray):
        videoDirName = videoDirNameTimeFrame[0]
        timeFrames = videoDirNameTimeFrame[1]
        keyFrames = [int((timeFrames[0]*2)+1), int((timeFrames[1]*2) + 1)]

        for i in range(keyFrames[0], keyFrames[1]):
            filename = ".." + fileSeparator + "keyframes_database" + fileSeparator + videoDirName + fileSeparator + videoDirName + "_frame" + str(
                i) + ".jpg"
            img = cv2.imread(filename)
            color = ('b', 'g', 'r')
            for j, col in enumerate(color):
                histr = cv2.calcHist([img], [j], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.savefig(("video-00" if ((i + 1 - keyFrames[0])<10) else "video-0") + str(i + 1 - keyFrames[0]) + ".png")
            plt.close()

        if platform.system() == "Windows":
            p = subprocess.Popen(["convertToVideo.bat", "videoRGBMatch" + str(index + 1)])
        else:
            p = subprocess.Popen(["convertToVideo.sh", "videoRGBMatch" + str(index + 1)])
        p.communicate()
        p.wait()

    index = 1
    for filename in glob.glob(queryPath + "*.jpg"):
        img = cv2.imread(filename)
        color = ('b', 'g', 'r')
        for j, col in enumerate(color):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.savefig("video-00" + str(index) + ".png")
        index = index + 1
        plt.close()
    if platform.system() == "Windows":
        p = subprocess.Popen(["convertToVideo.bat", "queryRGBMatch"])
    else:
        p = subprocess.Popen(["convertToVideo.sh", "queryRGBMatch"])
    p.communicate()
    p.wait()

#getTop3MatchedVideosWithTimeFrame("first")
#generateVideoHistogram([ ("musicVideo",[1,6]) , ("flowers",[1.5,6.5]), ("sports",[9.5,14.5]) ] , "first")