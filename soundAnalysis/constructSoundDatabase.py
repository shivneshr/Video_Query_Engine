import librosa
import glob
import numpy as np
import json
import platform

musicVideos = {}
fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"

for audioPath in glob.glob(".."+fileSeparator+"music_database"+fileSeparator+"*.wav"):

    musicVideo = {}
    audioName = audioPath.split(fileSeparator)[-1]

    for i in np.arange(0.0, 15.1, 0.5):
        v, vsr = librosa.load(audioPath, offset=i, duration=5)
        vmfcc = librosa.feature.mfcc(v, vsr)
        musicVideo[i] = vmfcc.tolist()

    musicVideos[audioName] = musicVideo


temp = {}
file = open("musicDatabase.json",'w')
json.dump(musicVideos,file)
file.close()