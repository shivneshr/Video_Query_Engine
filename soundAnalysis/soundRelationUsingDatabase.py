import librosa
import matplotlib.pyplot as plt
from dtw import dtw
import numpy as np
import glob
import json
import platform
from numpy.linalg import norm
from subprocess import Popen

fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"

def __findBestMatchedAudio(queryPath):

    q, qsr = librosa.load(queryPath)
    qmfcc = librosa.feature.mfcc(q, qsr)

    file = open("musicDatabase.json", 'r')
    musicVideos = json.load(file)
    audioMinDiffs = {}
    audioBestRange = {}
    for audioPath in glob.glob(".." + fileSeparator + "music_database"+fileSeparator+"*.wav"):
        audioName = audioPath.split(fileSeparator)[-1]
        musicVideo = musicVideos[audioName]
        audioDiff = 100000000
        timeframe = []
        for i in np.arange(0.0, 15.1, 1.0):
            v, vsr = librosa.load(audioPath, offset=i, duration=5)
            vmfcc = np.array(musicVideo[str(i)])
            dist, cost, acc_cost, path = dtw(qmfcc.T, vmfcc.T, dist=lambda x, y: norm(x - y, ord=1))
            if dist < audioDiff:
                audioDiff = dist
                timeframe = [i, i + 5]
            print("Dist", dist,audioDiff)
        audioMinDiffs[audioName] = audioDiff
        audioBestRange[audioName] = timeframe
    sortedDiffs = [(k, audioMinDiffs[k]) for k in sorted(audioMinDiffs, key=audioMinDiffs.get, reverse=False)]
    best = {}
    print(sortedDiffs)
    for i in range(0, 3):
        timeFrame = audioBestRange[sortedDiffs[i][0]]
        best[i + 1] = {}
        best[i + 1]["name"] = sortedDiffs[i][0].split(".")[0]
        best[i + 1]["timeFrames"] = [timeFrame[0], timeFrame[1]]
    return best

#input
#queryAudioName Eg - "HQ4"
#output
#{1: {'name': 'musicvideo', 'timeFrames': [4.0, 9.0]}, 2: {'name': 'sports', 'timeFrames': [15.0, 20.0]}, 3: {'name': 'traffic', 'timeFrames': [11.0, 16.0]}}
def getTop3MathcedAudiosWithTimeFrame(queryAudioName):
    queryPath = ".." + fileSeparator + "music_query" + fileSeparator + queryAudioName + ".wav"
    return __findBestMatchedAudio(queryPath)

# input
# videoDirNameArray Eg - [ ("musicVideo",[1,6]) , ("flowers",[1.5,6.5]), ("sports",[9.5,14.5]) ]
# queryVideoDirName Eg - "first"
# mainType Eg - "CNN"
def generateAudioMfccImages(audioTimeFrameArray, queryAudioName, mainType):
    queryPath = ".." + fileSeparator + "music_query" + fileSeparator + queryAudioName + ".wav"
    q, qsr = librosa.load(queryPath)
    qmfcc = librosa.feature.mfcc(q, qsr)
    plt.plot(qmfcc)
    plt.savefig("queryMFCC.png")
    plt.close()
    for index, audioTimeFrame in enumerate(audioTimeFrameArray):
        audioName = audioTimeFrame[0]
        timeFrame = audioTimeFrame[1]
        audioPath = ".." + fileSeparator + "music_database" + fileSeparator + audioName + ".wav"
        v, vsr = librosa.load(audioPath,offset=timeFrame[0], duration=(timeFrame[1] - timeFrame[0]))
        vmfcc = librosa.feature.mfcc(v, vsr)
        plt.plot(vmfcc)
        plt.savefig(mainType + "_AUD_" + audioName + "_" + str(index+1) + ".png")
        plt.close()
    if platform.system() == "Windows":
        p = Popen("moveToUI.bat")
    else:
        p = Popen("moveToUI.sh")
    p.communicate()
    p.wait()

print(getTop3MathcedAudiosWithTimeFrame("first"))
generateAudioMfccImages([ ("musicVideo",[1,6]) , ("flowers",[1.5,6.5]), ("sports",[9.5,14.5]) ] , "first", "AUD")