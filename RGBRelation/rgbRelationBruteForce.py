import glob
import cv2
import matplotlib.pyplot as plt
import platform
import subprocess
import os
import sys

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
# output
# {1: {'name': 'musicvideo', 'timeFrames': [4.0, 9.0], 'keyFrames': [9, 17], 'matchPercentage': '99.9855173832883%'}, 2: {'name': 'movie', 'timeFrames': [1.5, 6.5], 'keyFrames': [4, 12], 'matchPercentage': '76.1068256730008%'}, 3: {'name': 'starcraft', 'timeFrames': [6.5, 11.5], 'keyFrames': [14, 22], 'matchPercentage': '71.72459734687709%'}}
def getTop3MatchedVideosWithTimeFrame(queryVideoDirName):
    queryPath = ".." + fileSeparator + "keyframes_query"+fileSeparator+ queryVideoDirName + fileSeparator
    bestMatch = __getMatchingFrames(queryPath, cv2.HISTCMP_CORREL)
    rankedOutput = []
    for i in range(1, 4):
        rankedOutput.append((bestMatch[i]['name'], bestMatch[i]['timeFrames']))
    cnn.generateVideoPlots(rankedOutput, queryVideoDirName, 'RGB')
    rgb.generateVideoHistogram(rankedOutput, queryVideoDirName, 'RGB')
    audio.generateAudioMfccImages(rankedOutput, queryVideoDirName, 'RGB')

# input
# videoDirNameArray Eg - [ ("musicVideo",[1,6]) , ("flowers",[1.5,6.5]), ("sports",[9.5,14.5]) ]
# queryVideoDirName Eg - "first"
# mainType Eg - "CNN"
def generateVideoHistogram(videoDirNameTimeFrameArray, queryVideoDirName, mainType):
    queryPath = ".." + fileSeparator + "keyframes_query" + fileSeparator + queryVideoDirName + fileSeparator
    for index, videoDirNameTimeFrame in enumerate(videoDirNameTimeFrameArray):
        videoDirName = videoDirNameTimeFrame[0]
        timeFrames = videoDirNameTimeFrame[1]
        keyFrames = [int((timeFrames[0]*2)+1), int((timeFrames[1]*2) - 1)]

        for i in range(keyFrames[0], keyFrames[1] + 1):
            filename = ".." + fileSeparator + "keyframes_database" + fileSeparator + videoDirName + fileSeparator + videoDirName + "_frame" + str(
                i) + ".jpg"
            img = cv2.imread(filename)
            color = ('b', 'g', 'r')
            for j, col in enumerate(color):
                histr = cv2.calcHist([img], [j], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.savefig((".."+fileSeparator+"RGBRelation"+fileSeparator+"video-00" if ((i + 1 - keyFrames[0])<10) else "video-0") + str(i + 1 - keyFrames[0]) + ".png")
            plt.close()

        if platform.system() == "Windows":
            subprocess.Popen(["..\\RGBRelation\\convertToVideo.bat", mainType + "_RGB_" + videoDirName + "_" + str(index + 1)])
        else:
            subprocess.Popen(["../RGBRelation/convertToVideo.bat", mainType + "_RGB_" + videoDirName + "_" + str(index + 1)])

    index = 1
    for filename in glob.glob(queryPath + "*.jpg"):
        img = cv2.imread(filename)
        color = ('b', 'g', 'r')
        for j, col in enumerate(color):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.savefig(".."+fileSeparator+"RGBRelation"+fileSeparator+"video-00" + str(index) + ".png")
        index = index + 1
        plt.close()
    if platform.system() == "Windows":
        subprocess.Popen(["..\\RGBRelation\\convertToVideo.bat", "queryRGBMatch"])
    else:
        subprocess.Popen(["../RGBRelation/convertToVideo.bat", "queryRGBMatch"])

#getTop3MatchedVideosWithTimeFrame("first")
#generateVideoHistogram([ ("musicVideo",[1,6]) , ("flowers",[1.5,6.5]), ("sports",[9.5,14.5]) ] , "first", "RGB")