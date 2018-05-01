import platform
import os
import sys
from subprocess import Popen

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

import Similarity_Pipeline as pipeline
import rgbRelationBruteForce as rgb
import soundRelationUsingDatabase as audio

if platform.system() == "Windows":
    Popen("clearOutput.bat")
else:
    Popen(["sh","clearOutput.sh"])

query = "HQ4.MP4"
pipeline.find_match_CNN(query)
rgb.getTop3MatchedVideosWithTimeFrame(query)
audio.getTop3MathcedAudiosWithTimeFrame(query)
