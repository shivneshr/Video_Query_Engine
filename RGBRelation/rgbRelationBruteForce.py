import glob
import cv2
import platform

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
        for i in range(1,20):
            videoKeyFrameNames = dir + dir.split(fileSeparator)[-2] + "_frame" + str(i) + ".jpg"
            videoHist[i] = findHist(videoKeyFrameNames)

        if diffMethod in [cv2.HISTCMP_CORREL,cv2.HISTCMP_INTERSECT]:
            reverse = True

        videoMaxSum = 0
        for i in range(0,16):
            sum = 0
            for j in range(1,5):
                sum += cv2.compareHist(queryHist[j], videoHist[i+j], diffMethod)
            if(sum > videoMaxSum):
                videoMaxSum = sum
                videoBestFrame = [i+1,i+4]

        videoSums[dir.split(fileSeparator)[-2]] = videoMaxSum
        videoKeyFrame[dir.split(fileSeparator)[-2]] = videoBestFrame

    sortedSums = [(k, videoSums[k]) for k in sorted(videoSums, key=videoSums.get, reverse=reverse)]

    print("Best match video : " + sortedSums[0][0])
    print("Best match keyframes : " + str(videoKeyFrame[sortedSums[0][0]]))


# Please give directory name of the query
queryDirName = "HQ2.MP4"
getMatchingFrames(".." + fileSeparator + "keyframes_query"+fileSeparator+ queryDirName + fileSeparator, cv2.HISTCMP_CORREL)