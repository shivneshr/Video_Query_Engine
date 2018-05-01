import os
import shutil
import numpy as np
import pickle
import matplotlib.pyplot as plt
from subprocess import Popen
import platform

fileSeparator = "/"
if platform.system() == "Windows":
    fileSeparator = "\\"


keyframePath_database = ".."+fileSeparator+"keyframes_database"+fileSeparator
keyframePath_query = ".."+fileSeparator+"keyframes_query"+fileSeparator


def loadPickleFiles(location):
	pickleContent = []
	with open(location, 'rb') as handle:
		pickleContent = pickle.load(handle)

	return pickleContent


def __generateVideoPlot(databaseVideoName, queryVideoName, segment, procname, rank):

	segment = segment - 1

	# CNN file
	database_CNN_Video_file = keyframePath_database + databaseVideoName +fileSeparator+ databaseVideoName +'.pickle'
	query_CNN_Video_file = keyframePath_query + queryVideoName +fileSeparator+ queryVideoName +'.pickle'

	# CNN Label file
	database_CNN_Label_file = keyframePath_database + databaseVideoName +fileSeparator+ databaseVideoName +'_label.pickle'
	query_CNN_Label_file = keyframePath_query + queryVideoName + fileSeparator + queryVideoName + '_label.pickle'


	CNN_DB_Video = loadPickleFiles(database_CNN_Video_file)
	CNN_Query_Video = loadPickleFiles(query_CNN_Video_file)
	CNN_DB_Label = loadPickleFiles(database_CNN_Label_file)
	CNN_Query_Label = loadPickleFiles(query_CNN_Label_file)

	ctr=1

	videoName = procname+'_'+'CNN_'+databaseVideoName+'_'+str(rank)+'.mp4'

	for index_d,index_q in zip(range(segment,segment+9),range(0,9)):

		db_label = [[label1[1],label2[1]] for label1,label2 in zip(CNN_DB_Label[index_d],CNN_Query_Label[index_q])]

		diff = np.absolute(CNN_DB_Video[index_d] - CNN_Query_Video[index_q])
		sum = np.sum(diff)

		fig, ax = plt.subplots()

		# Hide axes as we want to show only the tables
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)

		#
		clust_data = np.array(db_label)
		collabel = ("Database Video", "Query Video match(" + str(sum) + ")")
		ax.table(cellText=clust_data, colLabels=collabel, loc='center')

		num =''
		if ctr<=9:
			num = '00'+str(ctr)
		elif ctr<=99:
			num = '0'+str(ctr)
		else:
			num = str(ctr)
		plt.savefig('..'+fileSeparator+'CNNRelation'+fileSeparator+'raw'+fileSeparator+'video-'+num+'.jpg')
		plt.close()
		ctr+=1

	if platform.system() == "Windows":
		Popen([".."+fileSeparator+"CNNRelation"+fileSeparator+"convertCNNVideo.bat",videoName])
	else:
		Popen(['sh',".."+fileSeparator+"CNNRelation"+fileSeparator+ "convertCNNVideo.sh",videoName])
		# shutil.move(".."+fileSeparator+"CNNRelation"+fileSeparator+'raw/CNNMatch.mp4','../ui/CNN/videos/'+videoName)

def generateVideoPlots(videoTimeFrameArray, queryVideoName, mainType):
	for index, videoTimeFrame in enumerate(videoTimeFrameArray):
		videoDirName = videoTimeFrame[0]
		timeFrame = videoTimeFrame[1]
		__generateVideoPlot(videoDirName, queryVideoName, int((timeFrame[0]+0.5)*2) , mainType, index + 1)


