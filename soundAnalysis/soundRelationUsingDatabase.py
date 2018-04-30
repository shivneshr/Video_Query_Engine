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

def findBestMatchedAudio(queryPath):

    q, qsr = librosa.load(queryPath)
    qmfcc = librosa.feature.mfcc(q, qsr)

    file = open("musicDatabase.json", 'r')
    musicVideos = json.load(file)
    print(musicVideos["flowers.wav"])

    audioMinDiffs = {}
    audioBestRange = {}
    for audioPath in glob.glob(".."+ fileSeparator + "music_database"+fileSeparator+"*.wav"):
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
    return sortedDiffs[0][0], audioBestRange[sortedDiffs[0][0]]

queryAudioName = "HQ4"
queryPath = ".."+fileSeparator+"music_query"+fileSeparator+queryAudioName+".wav"
audioName, bestMatchTimeFrame = findBestMatchedAudio(queryPath)
# audioName = 'flowers.wav'
# bestMatchTimeFrame = [10.0,15.0]
print("Best matched audio : ", audioName)
print("Best matched timeFrame : ", bestMatchTimeFrame)

q, qsr = librosa.load(queryPath)
qmfcc = librosa.feature.mfcc(q, qsr)
plt.plot(qmfcc)
plt.savefig("queryMFCC.png")
plt.close()

v,vsr = librosa.load(".."+fileSeparator+"music_database"+fileSeparator+audioName,offset=bestMatchTimeFrame[0], duration=(bestMatchTimeFrame[1]-bestMatchTimeFrame[0]))
vmfcc = librosa.feature.mfcc(v, vsr)
plt.plot(vmfcc)
plt.savefig("videoMFCC.png")
plt.close()

if platform.system() == "Windows":
    p = Popen("moveToUI.bat")
else:
    p = Popen("moveToUI.sh")
p.communicate()