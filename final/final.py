import platform
import os
import sys
from subprocess import Popen
import glob

fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"

if len(sys.argv) != 3:
    print("Give correct parameters")
    exit()

# User defined functions

UI_CNN_ScriptPath = ".."+fileSeparator+"CNNRelation"+fileSeparator
UI_RGB_ScriptPath = ".."+fileSeparator+"RGBRelation"+fileSeparator
UI_AUD_ScriptPath = ".."+fileSeparator+"soundAnalysis"+fileSeparator

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(UI_CNN_ScriptPath))
sys.path.append(os.path.abspath(UI_RGB_ScriptPath))
sys.path.append(os.path.abspath(UI_AUD_ScriptPath))

import Similarity_Pipeline as pipeline
import rgbRelationBruteForce as rgb
import soundRelationUsingDatabase as audio

inputType = sys.argv[1]

if inputType == "video":
    query = sys.argv[2]
    if platform.system() == "Windows":
        p = Popen(["copyQueryVideo.bat",query])
    else:
        p = Popen(["sh", "copyQueryVideo.sh",query])
elif inputType == "folder":
    queryFolder = sys.argv[2]
    for filename in glob.glob(queryFolder + fileSeparator + "*.wav"):
        query = filename.split(fileSeparator)[-1].split(".")[0]
    if platform.system() == "Windows":
        p = Popen(["createQueryVideo.bat",queryFolder,query])
    else:
        p = Popen(["sh", "createQueryVideo.sh",queryFolder,query])
else:
    exit()

p.wait()
print(sys.argv[0])
print(sys.argv[1])
print(sys.argv[2])


if platform.system() == "Windows":
    Popen("clearOutput.bat")
else:
    Popen(["sh","clearOutput.sh"])



keyframePath_query = ".."+fileSeparator+"keyframes_query"+fileSeparator
databasePath = ".."+fileSeparator+"videos"+fileSeparator+"query"+fileSeparator

# Extract key frames for the given query video
pipeline.extractKeyFrames_Second(databasePath+query,query,keyframePath_query)

# Extract the CNN features for the keyframes exracted
pipeline.extract_CNN_features(keyframePath_query,keyframePath_query+query)

# Find match based on CNN and generate match metrics
pipeline.find_match_CNN(query)

# Find match based on RGB and generate match metrics
rgb.getTop3MatchedVideosWithTimeFrame(query)

# Find match based on Audio and generate match metrics
audio.getTop3MathcedAudiosWithTimeFrame(query)
