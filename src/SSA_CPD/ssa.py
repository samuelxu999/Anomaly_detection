'''
========================
ssa.py
========================
Created on Dec.17, 2021
@author: Xu Ronghua
@Email:  rxu22@binghamton.edu
@TaskDescription: This module provide Singluar Spectrum Analysis functions for change point detection .
@Reference: 
'''

import logging
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)

def _create_hankel(ts_vect, lag_length, hankel_order, start_id):
	'''Create Hankel matrix.
	input
		ts_vect: 				full time series vector
		lag_length (M):			length of a column vector in Hankel matrix
		hankel_order (Q): 		order of Hankel matrix
		start_id : 				start index of ts_vect
	returns
		hankel_X:				hankel matrix, 2d array shape (lag_length, hankel_order)
	'''
	end_id = start_id + hankel_order
	hankel_X = np.empty((lag_length, hankel_order))
	for i in range(hankel_order):
	    hankel_X[:, i] = ts_vect[(start_id + i):(start_id + i + lag_length)]
	return hankel_X

def _score_svd(hankel_base, hankel_test, n_eofs):
	''' Run singular spectrum analysis algorithm to get change point detection scores
	input
		hankel_base: 			base matrix X_B
		hankel_test:			test matrix X_T
		n_eofs (I): 			n components for EOFs to build l-subspace L 
	returns
		score:					difference of the largest eigenvalue between U_test and U_base
	'''
	## apply svd to decompose hankel_base
	U_base, _, _ = np.linalg.svd(hankel_base, full_matrices=False)

	## apply svd to decompose hankel_test
	U_test, _, _ = np.linalg.svd(hankel_test, full_matrices=False)

	## perform the svd of lag-covariance matrix R=X*X.T 
	## use a particular group I including sorted n_eofs eigenvectors.
	## s is a list of scores that indicate differences of l largest eigenvalues
	_, s, _ = np.linalg.svd(U_test[:, :n_eofs].T @
	    U_base[:, :n_eofs], full_matrices=False)

	## return the score that use the difference of the largest eigenvalue.
	return 1 - s[0]

def _Edist_svd(hankel_base, hankel_test, n_eofs):
	'''Run svd to get Euclidean distances of hankel_base and hankel_test
	input
		hankel_base: 			base matrix X_B
		hankel_test:			test matrix X_T
		n_eofs (I): 			n components for EOFs to build l-subspace L 
	returns
		Euc_dist:				Euclidean distance between hankel_test and l-subspace
	'''
	## apply svd to decompose hankel_base, then use U_base to represent l-subspace
	U_base, _, _ = np.linalg.svd(hankel_base, full_matrices=False)

	## a) calculate Euclidean distance between test matrix and l-subspace matrix
	Euc_dist = np.linalg.norm(hankel_test.T @ hankel_test - 
		hankel_test.T @ U_base[:, :n_eofs] @ U_base[:, :n_eofs].T @ hankel_test)

	# Euc_dist = np.linalg.norm(hankel_test)**2 - np.linalg.norm(U_base[:, :n_eofs].T @ hankel_test)**2

	## return the Euclidean distance.
	return Euc_dist


class SingularSpectrumAnalysis():
	'''
	Singular Spectrum Analysis class to support change point detection.
	'''
	def __init__(self, lag_length, n_eofs=5, hankel_order=None, test_lag=None):
		''' Initialize parameters
		lag_length (M): int
	        The window length of a column vector in Hankel matrix. 
	        This also specify the length of a column vecter X_i of Hankel matrix
		n_eofs (I): int
	        n components for EOFs, which is used to build l-subspace, l<M.
	        This also specify how many rank of Hankel matrix will be taken
		hankel_order (Q): int
	        Hankel matrix length (K or Q). In general Q<=M. 
	        This also specify the number of column vecter X_i of Hankel matrix.  
		test_lag (p): int
	        The location of test matrix that is later than base matrix. In general p>M/2.   
		'''
		self.lag_length = lag_length
		self.n_eofs = n_eofs
		self.hankel_order = hankel_order
		self.test_lag = test_lag

	def Dn_Edist(self, ts_vect, scaled=False):
		''' Calculate squired Euclidean distances Di between test matrix and l-subspace
		input
			ts_vect:			full time serial vector
		returns
			Dn: 				list of squired Euclidean distances of test matrix and l-subspace.
		'''
		if self.hankel_order is None:
			# rule of thumb
			self.hankel_order = self.lag_length
		if self.test_lag is None:
			# rule of thumb
			self.test_lag = self.hankel_order // 2

		assert isinstance(ts_vect, np.ndarray), "input array must be numpy array."
		assert ts_vect.ndim == 1, "input array dimension must be 1."
		assert isinstance(self.lag_length, int), "window length must be int."
		assert isinstance(self.n_eofs, int), "number of components must be int."
		assert isinstance(self.hankel_order, int), "order of partial time series must be int."
		assert isinstance(self.test_lag, int), "lag between test series and history series must be int."

		## ts normalization (optional)
		if(scaled):
			ts_scaled = MinMaxScaler(feature_range=(1, 2))\
						.fit_transform(ts_vect.reshape(-1, 1))[:, 0]
		else:
			ts_scaled = ts_vect

		## initialize a list Dn, which is used to save squired Euclidean distances of test matrix and l-subspace Di
		Dn = np.zeros_like(ts_scaled)

		## set parameters.
		ts_size = ts_scaled.size
		M = self.lag_length
		Q = self.hankel_order
		p = self.test_lag

		end_point = ts_size - M - Q - p + 1

		for tid in range(1, end_point):
			## get base Hankel matrix
			base_id = tid -1
			hankel_base = _create_hankel(ts_scaled, M, Q, base_id)

			## get test Hankel matrix
			test_id = tid + p -1
			hankel_test = _create_hankel(ts_scaled, M, Q, test_id)
			
			## Set start id of Di
			D_id = tid -1
			# Dn[D_id] = _score_svd(hankel_base, hankel_test, self.n_eofs)
			Dn[D_id] = _Edist_svd(hankel_base, hankel_test, self.n_eofs)
		return Dn

	def Sn_norm(self, Dn):
		''' Calculate the normalized sum of squired distances between test matrix and l-subspace
		input
			Dn:					list of squired distances Di between test matrix and l-subspace
		returns
			Sn: 				list of normalized sum of squired distances.
		'''
		## initialize a list Sn, which is used to save the normalized sum of squired Euclidean distances Si
		Sn = np.zeros_like(Dn)

		## set parameters.
		ts_size = Dn.size
		M = self.lag_length
		Q = self.hankel_order
		p = self.test_lag

		end_p = ts_size - M - Q - p + 1

		## ------ calculate mu of Dn ????? ----------
		# mu_D = np.mean(Dn[:end_p])/(M*Q)
		## we use fixed mu_D to tolerant noisy
		mu_D = (M+p)/(M*Q)

		for tid in range(1, end_p):
			## set start and end id of Dn
			start_id = tid -1
			end_id = start_id + Q 

			## the sum of squired distances is normalized to the number of elements in the test matrix  
			# norm_D = np.mean(Dn[start_id:end_id])/(M*Q)
			norm_D = Dn[start_id]/(M*Q)

			## calculate normalized sum of squired distances Si
			if(mu_D>0.0):
				Si = norm_D/mu_D
			else:
				Si = norm_D

			## set start id of Si
			S_id = tid -1

			Sn[S_id] = Si

		return Sn

	def Wn_CUSUM(self, Sn):
		''' Calculate CUSUM statistics W
		input
			Sn:					list of normalized sum of squired distances
		returns
			W: 					list of CUSUM W given Sn.
		'''
		## initialize a list W, which is used to save the cumulative sum of Wi 
		W = np.zeros_like(Sn)

		## set parameters.
		ts_size = Sn.size
		M = self.lag_length
		Q = self.hankel_order
		p = self.test_lag

		## calculate CUSUM W
		W[0] = Sn[0]
		for tid in range(1, ts_size):
				temp_W = (W[tid-1] + Sn[tid]-Sn[tid-1]-1/(3*M*Q))
				W[tid] = max(0, temp_W)
		
		## calculate decision threshold
		# t_alpha = 1.6973	## alpha = 0.05, n=30
		t_alpha = 1.6839	## alpha = 0.05, n=40
		h = (2*t_alpha/(M*Q)) * math.sqrt((Q/3)*(3*M*Q-Q*Q+1))

		## shift points of W left to align the change points
		shift_pos = M + Q + p
		shift_W = np.zeros_like(W)
		shift_W[shift_pos:]=W[:ts_size-shift_pos]

		return shift_W, h

	def reconstruct(self, ts_vect, scaled=False):
		''' Reconstruct ts given original ts_vect with length lag_length
		input
			ts_vect:			full time serial vector
		returns
			recon_ts: 			rebuild ts vector
		'''
		if self.hankel_order is None:
			# rule of thumb
			self.hankel_order = self.lag_length

		assert isinstance(ts_vect, np.ndarray), "input array must be numpy array."
		assert ts_vect.ndim == 1, "input array dimension must be 1."
		assert isinstance(self.lag_length, int), "window length must be int."
		assert isinstance(self.n_eofs, int), "number of components must be int."
		assert isinstance(self.hankel_order, int), "order of partial time series must be int."

		## ts normalization (optional)
		if(scaled):
			ts_scaled = MinMaxScaler(feature_range=(1, 2))\
						.fit_transform(ts_vect.reshape(-1, 1))[:, 0]
		else:
			ts_scaled = ts_vect

		## set parameters.
		M = self.lag_length
		Q = self.hankel_order
		sval_nums = self.n_eofs
		ts_size = M+Q

		## get original ts
		original_ts = ts_scaled[:ts_size]

		## initialize recon_ts to save rebuild ts vector
		recon_ts = np.zeros_like(original_ts)

		## get Hankel matrix of original_ts
		hankel_ts = _create_hankel(original_ts, M, Q, 0)

		## apply svd to decompose hankel_ts
		U_ts, Sigma_ts, V_ts = np.linalg.svd(hankel_ts, full_matrices=False)
		
		## rebuild recon_hankel by using n_eofs singuar vectors
		recon_hankel = (U_ts[:,:sval_nums]).dot(np.diag(Sigma_ts[:sval_nums])).dot(V_ts[:sval_nums,:])
		
		## reconstruct ts
		for idx in range(Q):
			recon_ts[(idx):(idx + M)] = recon_hankel[:, idx]
		
		## use last point original_ts to iput last point of recon_ts
		recon_ts[-1] = original_ts[-1]
		return recon_ts

	@staticmethod
	def create_hankel(ts_vect, lag_length, hankel_order, start_id):
		'''Create Hankel matrix.
		input
			ts_vect: 				full time series vector
			lag_length (M):			length of a column vector in Hankel matrix
			hankel_order (Q): 		order of Hankel matrix
			start_id : 				start index of ts_vect
		returns
			hankel_X:				hankel matrix, 2d array shape (lag_length, hankel_order)
		'''
		end_id = start_id + hankel_order
		hankel_X = np.empty((lag_length, hankel_order))
		for i in range(hankel_order):
		    hankel_X[:, i] = ts_vect[(start_id + i):(start_id + i + lag_length)]
		return hankel_X
	
	@staticmethod
	def sigma_svd(hankel_matrix, n_eofs):
		_, sigma, _ = np.linalg.svd(hankel_matrix, full_matrices=False)
		return sigma[:n_eofs] 


