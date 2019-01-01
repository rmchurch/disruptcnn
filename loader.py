#!/usr/bin python

import numpy as np
import torch.utils.data as data
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler

class EceiDataset(data.Dataset):
	"""ECEi dataset"""

	def __init__(self,root,clear_file,disrupt_file,
					  train=True,flattop_only=True,
					  Twarn=300,Nmodel=300000,Nseq=300000):
		"""Initialize
		root: Directory root. Must have 'disrupt/' and 'clear/' as subdirectories
		clear_file, disrupt_file: File paths for disrupt/clear ECEi datasets.
			The files have the following columns (in order):
			  Shot - shot number
			  segments - number of ECEi segments (not needed)
			  tstart [ms] - start time
			  tlast [ms] - last time 
			  dt [ms] - timestep
			  SNR min - Minimum SNR of entire signal
			  t_flat_start [ms] - start time of flattop
			  t_flat_duration [ms] - time duration of flattop
			  tdisrupt [ms] - Time of disruption (corresponds to max dIp/dt)
		train: 
		flattop_only: Use only flattop times
		Twarn: time before disruption we wish to begin labelling as "disruptive"
		Nmodel: Number of time points used in neutral network to produce single timepoint prediction
		Nseq: Number of time points to divide data into (set by GPU memory constraints)
		"""
		self.root = root
		self.train = train #training set or test set TODO: Do I need this?

		self.Twarn = Twarn #in ms

		assert Nseq>=Nmodel, "Input Nseq has to be larger than or equal to Nmodel"
		self.Nseq = Nseq
		self.Nmodel = Nmodel

		data_disrupt = np.loadtxt(disrupt_file,skiprows=1)
		data_clear = np.loadtxt(clear_file,skiprows=1)
		data_all = np.vstack((data_disrupt,data_clear))

		tflatstarts = data_all[:,-3]
		if (np.any(np.isnan(tflatstarts))) and flattop_only: #NaN's used when no flattop in shot
			data_all = data_all[~np.isnan(tflatstarts)]
		tflatstarts = data_all[:,-3]
		tflatstops = data_all[:,-3] + data_all[:,-2]
		tstarts = data_all[:,2]
		tstops = data_all[:,3]
		dt = data_all[:,4]
		tdisrupt = data_all[:,-1]

		if flattop_only:
			self.start_idx = np.ceil((tflatstarts-tstarts)/dt).astype(int)
			self.stop_idx = np.floor((tflatstops-tstarts)/dt).astype(int)
		else:
			self.start_idx = np.ceil((0.-tstarts)/dt).astype(int)
			self.stop_idx = np.floor((tstops-tstarts)/dt).astype(int)

		#start_idx < idx < disrupt_idx: non-disruptive
		#disrupt_idx < idx < stop_idx: disruptive
		self.disrupt_idx = np.ceil((tdisrupt-self.Twarn-tstarts)/dt)
		self.disrupt_idx[tdisrupt<0] = np.nan #TODO: should this be something else? To allow int?

		self.shot = data_all[:,0].astype(int)

		#TODO: how to split? Need to know model length (determines overlap), and how long
		#can fit into GPU
		self.shots2seqs()

		self.calc_label_weights()

		self.length = len(self.start_idxi)



	def shots2seqs(self):
		"""Separate each shot into sequences (generates indices)"""
		#this is really the number of sequences, just results in too many seqs when Nseq~Nmodel
		#Nlong = self.stop_idx - self.start_idx + 1
		#num_seq_frac = (Nlong - self.Nseq + 1)/(self.Nseq - self.Nmodel + 1)
		#num_seq = np.ceil(num_seq_frac).astype(int)
		#this was the old number of sequences, not correct
		num_seq = np.ceil((self.stop_idx - self.start_idx + 1) / float(self.Nseq)).astype(int)
		
		self.start_idxi = []
		self.stop_idxi = []
		self.disrupt_idxi = []
		self.shoti = []
		self.shoti_type = []

		for i,(starti,stopi) in enumerate(zip(self.start_idx,self.stop_idx)):
			for m in range(num_seq[i]):
				self.start_idxi += [m*self.Nseq-m*self.Nmodel+m+starti]
				self.stop_idxi += [(m+1)*self.Nseq-m*self.Nmodel+m+starti]
				self.shoti += [self.shot[i]]
				#handle partial length sequence at end. TODO: Do I need this?
				if self.stop_idxi[-1]>stopi:
					self.stop_idxi[-1] = stopi
				if ((self.stop_idxi[-1] - self.start_idxi[-1])<self.Nmodel):
					self.start_idxi[-1] = self.stop_idxi[-1] - self.Nmodel

				if (self.disrupt_idx[i]>=self.start_idxi[-1]) and (self.disrupt_idx[i]<=self.stop_idxi[-1]):
					self.disrupt_idxi += [self.disrupt_idx[i]]
				else:
					self.disrupt_idxi += [np.nan]

				if np.isnan(self.disrupt_idx[i]):
					self.shoti_type += ['clear']
				else:
					self.shoti_type += ['disrupt']

	def calc_label_weights(self):
		""""""

	def train_val_test_split(self,sizes=[0.8,0.1,0.1]):
		"""Creates indices to split data into train, validation, and test datasets. 
		Stratifies to ensure each group has class structure consistent with original class balance
		sizes: Fractional size of train, validation, and test data sizes (3 element array or list)
		"""
		assert(np.array(sizes).size==3)
		assert(np.isclose(np.sum(sizes),1.0))

		#TODO: make labels based on each point, NOT just whether has disruptive points?
		labels = ~np.isnan(disrupt_idxi)

		self.train_indsi,valtest_inds,train_labels,valtest_labels = train_test_split(np.arange(self.start_idxi.size),labels,stratify=labels,test_size=np.sum(sizes[1:]))
		self.val_indsi, self.test_indsi, _, _ = train_test_split(valtest_inds,valtest_labels,stratify=valtest_labels,test_size=sizes[2]/np.sum(sizes[1:]))

	def __len__(self):
		"""Return total number of sequences"""
		return self.length

	def __getitem__(self,index):
		"""TODO, read the data from file"""
		f = h5py.File(self.root+
					  self.shoti_type[index]+'/'+
					  str(self.shoti[index])+'.h5')
		X = f['LFS'][...,self.start_idxi[index]:self.stop_idxi[index]]
		f.close()
		y = np.zeros(X.shape)

		return X,y


def data_generator(dataset,batch_size):
	"""Generate the loader objects for the train,validate, and test sets"""

	train_sampler = SubsetRandomSampler(dataset.train_indsi)
	valid_sampler = SubsetRandomSampler(dataset.val_indsi)
	test_sampler = SubsetRandomSampler(dataset.test_indsi)
	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
	                                           sampler=train_sampler)
	valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       sampler=valid_sampler)
	test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
	                                           sampler=test_sampler)
	return train_loader,val_loader,test_loader
