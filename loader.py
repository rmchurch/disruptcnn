#!/usr/bin python

import numpy as np
import torch.utils.data as data

class EceiDataset(data.Dataset):
	"""ECEi dataset"""

	def __init__(self,disrupt_file,clear_file,
					  train=True,flattop_only=True,
					  Twarn=300,Nmodel=300000,Nseq=300000):
		"""Initialize
		disrupt_file, clear_file: File paths for disrupt/clear ECEi datasets.
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
		self.train = train #training set or test set

		self.Twarn = Twarn #in ms

		assert Nseq>=Nmodel, "Input Nseq has to be larger than Nmodel"
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
		self.split_shots()

	def split_shots(self):
		num_seq = np.ceil((self.stop_idx - self.start_idx + 1) / float(self.Nseq)).astype(int)
		self.start_idxi = []
		self.stop_idxi = []
		self.shoti = []

		for i,(starti,stopi) in enumerate(zip(self.start_idx,self.stop_idx)):
			self.start_idxi += [m*self.Nseq-m*self.Nmodel+m+starti for m in range(num_seq[i])]
			self.stop_idxi += [(m+1)*self.Nseq-m*self.Nmodel+m+starti for m in range(num_seq[i])]
			self.shoti += [self.shot[i] for m in range(num_seq[i])]
			#handle partial length sequence at end. TODO: Do I need this?
			if self.stop_idxi[-1]>stopi: self.stop_idxi[-1] = stopi





	def __len__(self):
		"""TODO"""

	def __getitem__(self,index):
		"""TODO, read the data from file"""
		f = h5py.File(self.root_dir+
					  self.shot_type[index]+'/'+
					  self.shots[index]+'.h5')
		X = f['LFS'][self.startinds[index]:self.stopinds[index]]
		f.close()
		y = self.labels[index]

		return X,y