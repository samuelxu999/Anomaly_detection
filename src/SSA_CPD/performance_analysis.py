'''
========================
Performance measurement and analysis module
========================
Created on Feb.2, 2022
@author: Xu Ronghua
@Email:  rxu22@binghamton.edu
@TaskDescription: This module provide performance measurement utilities.
'''
import sys
import logging
import argparse

import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
from utils.utilities import FileUtil
import numpy as np
# import math

'''
Data preparation class
''' 
class PreData(object):
	@staticmethod
	def avg_exec_time(exec_time_csv): 
		'''
		Function: load data from csv file and calculate average of each column.
		@(in):		csv file that save exec time
		@(out):		return avarage of columns
		''' 

		## used to save mean of column vector
		ls_avg_exec_time = []

		## load data to np_exec_time
		np_exec_time = FileUtil.read_csv(exec_time_csv)
		## for each column to get mean of vector
		for col_vect in np_exec_time.T:
			ls_avg_exec_time.append(np.mean(col_vect))

		return ls_avg_exec_time

	@staticmethod
	def load_cpu_memory(top_log): 
		'''
		Function: load data from csv file and calculate average of each column.
		@(in):		log file that save resource usage
		@(out):		return list format of cpu and memory data
		''' 

		## used to save mean of column vector
		ls_avg_exec_time = []

		## 1) load data to np_exec_time
		np_log_data = FileUtil.read_top_log(top_log)

		## 2) remove space cell
		ls_data_nospace = []
		for row in np_log_data:
			row_no_space = []
			for cell in row:
				if(cell!=''):
					row_no_space.append(cell)
			ls_data_nospace.append(row_no_space)


		## 3) get cpu and memory column
		np_log_data_nosapce = np.array(ls_data_nospace)

		np_cpu_memory = np_log_data_nosapce[:,-4:-2]

		return [np_log_data_nosapce[:,-4].tolist(), np_log_data_nosapce[:,-3].tolist()]

	@staticmethod
	def error_statistics(ls_vector): 
		np_data=np.array(ls_vector, dtype=np.float32)

		ave_exec_time=format(np.average(np_data), '.3f' )
		median_exec_time=format(np.median(np_data), '.3f' )
		std_exec_time=format(np.std(np_data), '.3f' )
		max_exec_time=format(np.max(np_data), '.3f' )
		min_exec_time=format(np.min(np_data), '.3f' )

		return [ave_exec_time, std_exec_time, median_exec_time, max_exec_time, min_exec_time]		

'''
Data visualization class to display data as bar or lines
''' 
class VisualizeData(object):
	@staticmethod
	def autolabel(rects, ax):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2, (height+0.2),
				'%.1f' % height,
				ha='center', va='bottom', fontsize=12)

	@staticmethod
	def groupbars_platform(xtick_label, y_label, legend_label, ls_data):
		Y_RATIO = 1
		N = len(xtick_label)

		ind = np.arange(N)  # the x locations for the groups
		width = 0.25           # the width of the bars

		#generate bar axis object
		fig, ax = plt.subplots()

		ls_color=['r', 'royalblue', 'forestgreen']

		rects_bar = []
		idx = 0
		for avg_exec_time in ls_data:
			rects_bar.append(ax.bar(ind + width*idx, avg_exec_time, width, color=ls_color[idx]))
			idx+=1

		## add some text for labels, title and axes ticks
		ax.set_ylabel(y_label, fontsize=16)
		#ax.set_title('Execution time by group', fontsize=18)
		ax.set_xticks(ind + width)
		ax.set_xticklabels(xtick_label, fontsize=16)
		# plt.ylim(0, 220)

		# ax.legend((rects_tx[0], rects_block[0], rects_vote[0]), legend_label, loc='upper left', fontsize=18)
		ax.legend(rects_bar, legend_label, loc='upper center', fontsize=18)

		idx = 0
		for avg_exec_time in ls_data:
			VisualizeData.autolabel(rects_bar[idx], ax)
			idx+=1
		plt.show()

	@staticmethod
	def scatter_platform(x_label, y_label, legend_label, ls_data):
		xtick_len = min(len(ls_data[0]),len(ls_data[1]))

		## new x list
		x_list = list(range(1, xtick_len+1))

		ls_marker =['.', '*', 'o', '^', '>']
		ls_color=['r', 'darkorange', 'seagreen']

		## for each vector to plot line
		line_list=[]
		idx=0
		for cpu_data in ls_data:
			np_cpu = np.array(cpu_data[:xtick_len], dtype=np.float32)
			line_list.append(plt.plot(x_list, np_cpu, linestyle="", marker=ls_marker[idx], color=ls_color[idx]))
			idx+=1

		plt.xlabel(x_label, fontsize=16)
		plt.ylabel(y_label, fontsize=16)
		plt.ylim(0, 130)
		# plt.title(title_name)
		plt.legend(legend_label, loc='upper left', fontsize=18)

		#show plot
		plt.show()

	'''
	plot errror bars shown mdedian and std given ls_dataset[mean, std, median, max, min]
	'''
	@staticmethod
	def plot_errorBar(legend_label, ax_label, ls_dataset):

		N = len(legend_label)

		# the x locations for the groups
		ind = np.arange(N)

		np_dataset=np.array(ls_dataset, dtype=np.float32)
		trans_np_dataset=np_dataset.transpose()
		ls_mean = trans_np_dataset[0]
		ls_std = trans_np_dataset[1]
		ls_median = trans_np_dataset[2]
		ls_max = trans_np_dataset[3]
		ls_min = trans_np_dataset[4]

		fig, ax = plt.subplots()

		# create stacked errorbars:
		plt.errorbar(ind, ls_mean, ls_std, fmt='or', ecolor='seagreen', lw=30)
		plt.errorbar(ind, ls_median, [ls_mean - ls_min, ls_max - ls_mean], 
					fmt='*k', ecolor='gray', lw=5)

		ax.set_xticks(ind)
		ax.set_xticklabels(legend_label, fontsize=18)
		ax.set_ylabel(ax_label[1], fontsize=18)
		ax.yaxis.grid(True)
		plt.xlim(-0.5, 2.5)

		plt.show()


def plot_avg_exec_time(args):
	ls_platform = ['Desktop','Rpi4','Rpi3']

	ls_avg_exec_time = []
	for device in ls_platform:
		file_name = 'test_results/' + device + '/exec_ssa_time1.csv'

		ls_avg_exec_time.append(PreData.avg_exec_time(file_name))
	# print(ls_avg_exec_time)


	xtick_label=['D', 'S', 'W', 'Total']
	y_label = 'Time (ms)'
	legend_label=['Desktop', 'Raspberry Pi 4', 'Raspberry Pi 3']

	VisualizeData.groupbars_platform(xtick_label, y_label, legend_label, ls_avg_exec_time)

def plot_cpu_memory(args):
	ls_platform = ['Desktop','Rpi4','Rpi3']

	ls_cpu = []
	ls_memory = []

	for device in ls_platform:
		if(args.interval==1):
			file_name = 'test_results/' + device + '/sys_top5.log'
		else:
			file_name = 'test_results/' + device + '/sys_top1.log'
		ls_cpu.append(PreData.load_cpu_memory(file_name)[0])
		ls_memory.append(PreData.load_cpu_memory(file_name)[1])

	## calculate average memory usage
	mem_size = [16, 4, 1]
	ls_avg_memory =[]
	idx = 0
	for memory_vector in ls_memory:	
		np_memory =np.array(memory_vector, dtype=np.float32)
		avg_memory = np.mean(np_memory)*mem_size[idx]*10
		ls_avg_memory.append(format(avg_memory, '.1f'))
		idx+=1
	
	print("Memory usage (MB): \t Desktop-{}\t Rpi4-{}\t Rpi3-{}\t".format(ls_avg_memory[0], 
													ls_avg_memory[2], ls_avg_memory[1]))


	x_label = 'Time sequentce'
	y_label = 'CPU usage (%)'
	legend_label=['Desktop', 'Raspberry Pi 4', 'Raspberry Pi 3']
	VisualizeData.scatter_platform(x_label, y_label, legend_label,ls_cpu)

def plot_resource_usage(args):
	ls_platform = ['Desktop','Rpi4','Rpi3']

	ls_cpu = []
	ls_memory = []
	for device in ls_platform:
		if(args.interval==1):
			file_name = 'test_results/' + device + '/sys_top5.log'
		else:
			file_name = 'test_results/' + device + '/sys_top1.log'
		
		cpu_data = PreData.load_cpu_memory(file_name)[0]
		ls_cpu.append(PreData.error_statistics(cpu_data))

	ax_label = ['', 'CPU usage (%)']
	legend_label=['Desktop (1.6 GHZ)', 'RPi 4 (1.5 GHZ)', 'RPi 3 (1.2 GHZ)']

	VisualizeData.plot_errorBar(legend_label, ax_label, ls_cpu)

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run test evaulation app."
    )
    parser.add_argument("--test_func", type=int, default=0, 
                        help="Execute test operation: \
                        0-function test, \
                        1-plot_avg_exec_time \
                        2-plot_cpu_memory \
                        3-plot_resource_usage")
    parser.add_argument("--interval", type=int, default=0, 
                        help="Sleep time between test round.: 0-1 sec, 1-5 sec")
    parser.add_argument("--platform", type=int, default=2, 
                        help="Test platform: 0-desktop, 1-Rpi4, 2-Rpi3,")
    args = parser.parse_args(args=args)
    return args

if __name__ == "__main__":

	args = define_and_get_arguments()

	if(args.test_func==1):
		plot_avg_exec_time(args)
	elif(args.test_func==2):
		plot_cpu_memory(args)
	elif(args.test_func==3):
		plot_resource_usage(args)
	else:
		file_name = 'test_results/' + 'Rpi4/' + 'sys_top1.log'
		ls_avg_exec_time = PreData.load_cpu_memory(file_name)
		print(ls_avg_exec_time)