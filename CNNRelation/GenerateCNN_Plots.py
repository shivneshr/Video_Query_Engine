import os
import shutil
import numpy as np
import pickle
import matplotlib.pyplot as plt
from subprocess import Popen


keyframePath_database = "/Users/shivnesh/Documents/shivnesh_git/multimedia_project/keyframes_database/"
keyframePath_query = "/Users/shivnesh/Documents/shivnesh_git/multimedia_project/keyframes_query/"


def loadPickleFiles(location):

	pickleContent = []
	with open(location, 'rb') as handle:
		pickleContent = pickle.load(handle)

	return pickleContent



def generateVideoPlots(databaseVideoName, queryVideoName, segment, procname,rank):

	# CNN file
	database_CNN_Video_file = keyframePath_database + databaseVideoName +'/'+ databaseVideoName +'.pickle'
	query_CNN_Video_file = keyframePath_query + queryVideoName +'/'+ queryVideoName +'.pickle'

	# CNN Label file
	database_CNN_Label_file = keyframePath_database + databaseVideoName +'/'+ databaseVideoName +'_label.pickle'
	query_CNN_Label_file = keyframePath_query + queryVideoName + '/' + queryVideoName + '_label.pickle'


	CNN_DB_Video = loadPickleFiles(database_CNN_Video_file)
	CNN_Query_Video = loadPickleFiles(query_CNN_Video_file)
	CNN_DB_Label = loadPickleFiles(database_CNN_Label_file)
	CNN_Query_Label = loadPickleFiles(query_CNN_Label_file)

	ctr=1

	videoName = procname+'_'+'CNN_'+databaseVideoName+'_'+str(rank)+'.mp4'

	for index_d,index_q in zip(range(segment,segment+9),range(0,9)):

		db_label = [[label1[1],label2[1]] for label1,label2 in zip(CNN_DB_Label[index_d],CNN_Query_Label[index_q])]
		np_db_label = np.array(db_label)

		#query_label = [label[1] for label in CNN_Query_Label[index]]

		diff = np.absolute(CNN_DB_Video[index_d] - CNN_Query_Video[index_q])
		sum = np.sum(diff)

		fig, ax = plt.subplots()
		# Hide axes
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)

		# Table from Ed Smith answer
		clust_data = np.array(db_label)
		#clust_data = np.random.random((10,2))
		collabel = ("Database Video", "Query Video match(" + str(sum) + ")")
		ax.table(cellText=clust_data, colLabels=collabel, loc='center')

		num =''
		if ctr<=9:
			num = '00'+str(ctr)
		elif ctr<=99:
			num = '0'+str(ctr)
		else:
			num = str(ctr)
		plt.savefig('raw/video-'+num+'.jpg')
		ctr+=1


	Popen(['sh','./convertCNNVideo.sh'])
	shutil.move('raw/CNNMatch.mp4','../ui/CNN/videos/'+videoName)

#Popen(['sh','./convertCNNVideo.sh'])


