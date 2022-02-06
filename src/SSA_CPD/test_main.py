'''
========================
test_main.py
========================
Created on Dec.17, 2021
@author: Xu Ronghua
@Email:  rxu22@binghamton.edu
@TaskDescription: This is used to unit function test and demo.
@Reference: 
'''

import sys
import time
import random
import logging
import argparse
from utils.utilities import FileUtil, TypesUtil, PlotUtil
from ssa import SingularSpectrumAnalysis

## use tkagg to remotely display plot
# import matplotlib
# matplotlib.use('tkagg')

logger = logging.getLogger(__name__)


def show_data(args):
	data_file = "./data/rho.pkl"

	rho = FileUtil.pkl_read(data_file)

	dataset = []
	ls_data = TypesUtil.np2list(rho)
	dataset.append(ls_data)

	fig_file = "Data_figure"
	PlotUtil.Plotline(dataset, is_show=args.show_fig, is_savefig=args.save_fig, datafile=fig_file)

def load_data():
	data_file = "./data/rho.pkl"

	ts_data = FileUtil.pkl_read(data_file)

	return ts_data
	 

def ssa_test(args):
	## load time series data
	ts_vector = load_data()
	# print(ts_vector.shape)

	## noisy tolerant 
	if(args.op_status==1):
		## inject noisy data
		ts_vector[150:151]=0.1
		ts_vector[200:201]=0.8
		ts_vector[201:202]=0.7
		ts_vector[202:203]=0.6
		ts_vector[203:204]=0.5
		ts_vector[204:205]=0.4
		ts_vector[350:352]=0.2
		ts_vector[450:451]=0.15	
	## attack detect
	elif(args.op_status==2):
		## inject noisy data
		ts_vector[150:151]=0.1
		ts_vector[450:451]=0.15	

		## inject fakedata
		ts_vector[200:225]=0.3
		ts_vector[330:340]=0.1		
	## normal data
	else:
		pass

	## start to count exe time
	start_time=time.time()
	## pre) initialize SSA object
	cpd_ssa = SingularSpectrumAnalysis(lag_length=40, n_eofs=5, test_lag=20, hankel_order=40)
	
	## 1) apply SSA to get Euclidean distances D
	D = cpd_ssa.Dn_Edist(ts_vector, scaled=True)

	## 2) get normalized sum of squired distances S.
	S = cpd_ssa.Sn_norm(D)

	## 3) calculate CUSUM statistics W
	W, h = cpd_ssa.Wn_CUSUM(S)

	## get exe time
	exec_time=time.time()-start_time
	print("SSA test running time: {:.3f} s".format(exec_time))

	## 4) plot ts data and scores
	PlotUtil.plot_data_and_score(ts_vector, W, h, args.show_fig)

def ssa_performance(args):
	## load time series data
	ts_vector = load_data()

	## use a section of ts_vector to performe ssa
	test_section = ts_vector[:300]

	## inject noisy data
	test_section[150:151]=0.1

	## inject fakedata
	test_section[200:225]=0.3

	ls_dataset = []	

	for x in range(args.test_round):
		logger.info("Test run:{}".format(x+1))

		ls_time = []

		## pre) initialize SSA object
		cpd_ssa = SingularSpectrumAnalysis(lag_length=40, n_eofs=5, test_lag=20, hankel_order=40)

		## start to count exe time
		start_time=time.time()	
		## 1) apply SSA to get Euclidean distances D
		start_stage=time.time()
		D = cpd_ssa.Dn_Edist(test_section, scaled=True)
		exec_time=time.time()-start_stage
		ls_time.append(format(exec_time*1000, '.3f'))

		start_stage=time.time()
		## 2) get normalized sum of squired distances S.
		S = cpd_ssa.Sn_norm(D)
		exec_time=time.time()-start_stage
		ls_time.append(format(exec_time*1000, '.3f'))

		start_stage=time.time()
		## 3) calculate CUSUM statistics W
		W, h = cpd_ssa.Wn_CUSUM(S)
		exec_time=time.time()-start_stage
		ls_time.append(format(exec_time*1000, '.3f'))

		## get exe time
		exec_time=time.time()-start_time
		ls_time.append(format(exec_time*1000, '.3f'))

		ls_dataset.append(ls_time)

		time.sleep(args.wait_interval)

	## save log to local
	if(args.save_log):
		FileUtil.save_csv('test_results', 'exec_ssa_time', ls_dataset)		
	else:
		logger.info("SSA test running time (ms): {}\n".format(ls_time))

	## 4) plot ts data and scores
	if(args.show_fig):
		PlotUtil.plot_data_and_score(test_section, W, h, True)

def define_and_get_arguments(args=sys.argv[1:]):
	parser = argparse.ArgumentParser(description="Run test.")

	parser.add_argument("--test_func", type=int, default=0, 
						help="Execute test function: 0-show_data(), \
													1-ssa_test()")

	parser.add_argument("--op_status", type=int, default=0, help="test case type.")

	parser.add_argument("--show_fig", action="store_true", help="Show plot figure model.")

	parser.add_argument("--show_info", action="store_true", help="Print test information on screen.")

	parser.add_argument("--save_fig", action="store_true", help="Save plot figure on local disk.")

	parser.add_argument("--save_log", action="store_true", help="Save test logs on local disk.")

	parser.add_argument("--test_round", type=int, default=1, help="test evaluation round")

	parser.add_argument("--wait_interval", type=int, default=1, help="break time between test round.")

	args = parser.parse_args(args=args)
	return args

if __name__ == '__main__':
	FORMAT = "%(asctime)s %(levelname)s | %(message)s"
	LOG_LEVEL = logging.INFO
	logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

	# ssa_logger = logging.getLogger("ssa")
	# ssa_logger.setLevel(logging.DEBUG)

	args = define_and_get_arguments()

	if(args.test_func==1):
		ssa_test(args)
	elif(args.test_func==2):
		ssa_performance(args)
	else:
		show_data(args)
