import librosa
from dtw import dtw
import numpy as np
import glob
import platform
from numpy.linalg import norm

fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"

def findBestMatchedAudio(queryPath):

    q, qsr = librosa.load(queryPath)
    qmfcc = librosa.feature.mfcc(q, qsr)

    audioMinDiffs = {}
    audioBestRange = {}
    for audioPath in glob.glob(".."+ fileSeparator + "music_database"+fileSeparator+"*.wav"):
        audioName = audioPath.split(fileSeparator)[-1]
        audioDiff = 100000000
        timeframe = []
        for i in np.arange(0.0, 15.1, 1.0):
            v, vsr = librosa.load(audioPath, offset=i, duration=5)
            vmfcc = librosa.feature.mfcc(v, vsr)
            dist, cost, acc_cost, path = dtw(qmfcc.T, vmfcc.T, dist=lambda x, y: norm(x - y, ord=1))
            if dist < audioDiff:
                audioDiff = dist
                timeframe = [i, i + 5]
            print("Dist", dist,audioDiff)
        audioMinDiffs[audioName] = audioDiff
        audioBestRange[audioName] = timeframe

    sortedDiffs = [(k, audioMinDiffs[k]) for k in sorted(audioMinDiffs, key=audioMinDiffs.get, reverse=False)]

    print("Best matched audio : ", sortedDiffs[0][0])
    print("Best matched timeFrame : ", audioBestRange[sortedDiffs[0][0]])

queryAudioName = "HQ4"
findBestMatchedAudio(".."+fileSeparator+"music_query"+fileSeparator+queryAudioName+".wav")