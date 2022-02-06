'''
========================
utilities.py
========================
Created on Dec.17, 2021
@author: Xu Ronghua
@Email:  rxu22@binghamton.edu
@TaskDescription: This module provide utility function to support project.
@Reference: 
'''

from datetime import datetime, timedelta
import os
import json
import numpy as np
import pickle
import csv
import matplotlib
import matplotlib.pyplot as plt

'''
FileUtil class for handling file data
'''
class FileUtil(object):
	@staticmethod
	def pkl_read(pkl_file):
		'''
		Function: Read data from pkl file
		@arguments: 
		(out) np_dataset:   	return np.array object
		(in) pkl_file:   		pkl file path
		'''
		handle_pkl_file = open(pkl_file, 'rb')
		_data = pickle.load(handle_pkl_file)
		handle_pkl_file.close()
		## convert to np array
		np_data = np.array(_data[0],dtype=np.float32)
		return np_data

	@staticmethod
	def save_csv(test_dir, log_file, ls_dataset):
		'''
		Function: Write array to csv file
		@arguments: 
		(in) test_dir:   	test result directory to save csv file
		(in) log_file:   	log file name
		(in) ls_dataset:   	list format of dataset
		'''
		## check if test_dir is availble
		if(not os.path.exists(test_dir)):
			os.makedirs(test_dir)

		## set log file path
		csv_file = test_dir + '/' + log_file + '.csv'

		## convert ls_dataset to np matrix
		np_data = np.array(ls_dataset, dtype=np.float32)

		## for each row to write csv file
		with open(csv_file, 'w') as csv_handle:
			csv_writer = csv.writer(csv_handle, delimiter=',')
			for row in np_data:
				csv_writer.writerow(row)

	@staticmethod
	def read_csv(csv_file):
		'''
		Function: Read data from csv file
		@arguments: 
		(out) np_dataset:   	return np.array(float32) object
		(in) csv_file:   		csv file path
		'''
		ls_dataset = []
		with open(csv_file, 'r') as csvFile:
			csv_reader = csv.reader(csvFile, delimiter=',')
			for row in csv_reader:
				ls_dataset.append(row)
		# transfer to np array and return
		np_dataset = np.array(ls_dataset, dtype=np.float32)
		return np_dataset

	@staticmethod
	def read_top_log(csv_file):
		'''
		Function: Read top log data from csv file
		@arguments: 
		(out) np_dataset:   	return np.array(float32) object
		(in) csv_file:   		csv file path
		'''
		ls_dataset = []
		with open(csv_file, 'r') as csvFile:
			csv_reader = csv.reader(csvFile, delimiter=' ')
			for row in csv_reader:
				ls_dataset.append(row)
		# transfer to np array and return
		np_dataset = np.array(ls_dataset)
		return np_dataset
		
			

'''
TypesUtil class for data type format transfer
'''
class TypesUtil(object):
	# list dataset to numpy matrix
	@staticmethod
	def list2np(ls_data):
		# transfer to np array and return
		np_data = np.array(ls_data, dtype=np.float32)
		return np_data

	# numpy matrix to list dataset
	@staticmethod
	def np2list(np_data):
		# transfer to list dataset and return
		ls_data = np_data.tolist()
		return ls_data


'''
PlotUtil class for data visualization
'''
class PlotUtil(object):
	@staticmethod
	def Plotline(ENF_dataset, legend_label='', font_size=14, is_show=True, is_savefig=False, datafile=''):
		'''
		Function: plot ENF data as line on fig
		@arguments: 
		(in) ENF_dataset: 	list dataset that can input two ENF signals
			   font_size:	font size for label and legend
			     is_show:	Display plot on screen
			  is_savefig:	Save plot on local as *.png
			  	datafile:	file name to save plot
		'''
		ls_color=['g', 'seagreen', 'darkorange', 'r', 'b', 'gray']
		leg_label = []
		## For each node to get ENF vector
		for ENF_id in range(len(ENF_dataset)):
			#generate x and y data
			xdata = [];
			ydata = [];
			ls_dataset = ENF_dataset[ENF_id]
			## For each value in ENF vector to asssign <x,y>
			for i in range(0, len(ls_dataset)):
				xdata.append(i)
				ydata.append( float(ls_dataset[i]) )

			## plit line for ENF_id
			plt.plot(xdata, ydata, lw=2.0, color=ls_color[ENF_id])

			## add ENF_id for legend label
			leg_label.append("ENF-{}".format(ENF_id))

		## set x and y label text
		plt.xlabel('Time slot', fontsize=font_size)
		plt.ylabel('ENF coef', fontsize=font_size) 
		# plt.ylim(59.995, 60.005)

		## plot legend given legend label 
		if(legend_label == ''):
			plt.legend(leg_label, loc='best', fontsize=font_size)
		else:
			plt.legend(legend_label, loc='best', fontsize=font_size)
		
		## show figure if is_show is enabled
		if( is_show ):
			plt.show()

		## save figure if is_savefig is enabled
		if( is_savefig ):
			figname = os.path.splitext(datafile)[0] +'.png'
			plt.savefig(figname)
		plt.close()

	@staticmethod
	def plot_data_and_score(raw_data, score, threshold, is_show=False):
		f,ax = plt.subplots(2, 1, figsize=(20, 10))
		## plot raw data
		ax[0].plot(raw_data, lw=1.0, color='b')
		ax[0].set_ylabel('ENF coef score')
		ax[0].set_title("ENF coef data")

		## plot score and h
		h = np.zeros_like(score)
		h[:] = threshold
		ax[1].plot(score,lw=1.0, color='g')
		ax[1].plot(h,lw=1.0, color='r', linestyle='dashed')
		ax[1].set_ylabel('CUSUM W')
		ax[1].set_title("CUSUM-type W")
		
		## show figure if is_show is enabled
		if( is_show ):
			plt.show()

		## close plot
		plt.close()
