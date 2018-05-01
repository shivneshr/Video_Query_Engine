import platform
import json
from subprocess import Popen

fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"

def generateVideosForGivenTimeFrames(rankedOutput, mainType):
    jsonObj = {}
    for index, videoDirNameTimeFrame in enumerate(rankedOutput):
        jsonObj[index+1] = {}
        videoDirName = videoDirNameTimeFrame[0]
        jsonObj[index+1]['video'] = videoDirName
        timeFrames = videoDirNameTimeFrame[1]
        jsonObj[index+1]['time'] = timeFrames
        timeFrames[0] = int(timeFrames[0])
        timeFrames[1] = int(timeFrames[1])
        startTime = ("%02d" % timeFrames[0]) if timeFrames[0] < 10 else str(timeFrames[0])
        videoPath = ".."+fileSeparator+"videos"+fileSeparator+"search"+fileSeparator+videoDirName+".mp4"
        outputVideoName = ".."+fileSeparator+"ui"+fileSeparator + "videos" + fileSeparator + mainType+"_"+videoDirName+"_"+ str(index + 1) +".mp4"
        if platform.system() == "Windows":
            Popen(["..\\generateVideos\\generateTrimmedVideos.bat",videoPath,outputVideoName, startTime])
        else:
            Popen(["sh", "../generateVideos/generateTrimmedVideos.sh",videoPath,outputVideoName, startTime])

    file = open(".."+fileSeparator+"ui"+fileSeparator + mainType + ".json", 'w')
    json.dump(jsonObj,file)
    file.close()

#generateVideosForGivenTimeFrames([('musicvideo', [4.0, 9.0]), ('movie', [11.5, 16.5]), ('starcraft', [6.5, 11.5])], "RGB")