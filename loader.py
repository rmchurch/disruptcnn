#!/usr/bin python

import numpy as np
import torch.utils.data as data
import h5py
from sklearn.model_selection import train_test_split
#from torch.utils.data import SubsetRandomSampler

class EceiDataset(data.Dataset):
	"""ECEi dataset"""

	def __init__(self,root,clear_file,disrupt_file,
					  train=True,flattop_only=True,
					  Twarn=300):
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
			  tdisrupt [ms] - Time of disruption (corresponds to max dIp/dt). -1000 if not disrupted.
		train: 
		flattop_only: Use only flattop times
		Twarn: time before disruption we wish to begin labelling as "disruptive"
		"""
		self.root = root
		self.train = train #training set or test set TODO: Do I need this?

		self.Twarn = Twarn #in ms

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
		self.disrupt_idx = np.ceil((tdisrupt-self.Twarn-tstarts)/dt).astype(int)
		self.disrupt_idx[tdisrupt<0] = -1000 #TODO: should this be something else? nan isnt possible with int
		self.disrupted = self.disrupt_idx>0 #True if disrupted, False if not

		self.shot = data_all[:,0].astype(int)
		self.length = len(self.shot)

		#TODO: how to split? Need to know model length (determines overlap), and how long
		#can fit into GPU
		#No longer split into sequences at the Dataloader level, chunk later
		#self.shots2seqs()

		self.calc_label_weights()


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
		labels = self.disrupted

		self.train_inds,valtest_inds,train_labels,valtest_labels = train_test_split(np.arange(self.shot.size),labels,
																					stratify=labels,
																					test_size=np.sum(sizes[1:]))
		self.val_inds, self.test_inds, _, _ = train_test_split(valtest_inds,valtest_labels,
															   stratify=valtest_labels,
															   test_size=sizes[2]/np.sum(sizes[1:]))

	def __len__(self):
		"""Return total number of sequences"""
		return self.length

	def __getitem__(self,index):
		"""Read the data from file. Reads the entire sequence from the shot file"""
		
		if self.disrupted[index]:
			shot_type='disrupt'
		else:
			shot_type='clear'

		f = h5py.File(self.root+
					  shot_type+'/'+
					  str(self.shot[index])+'.h5')
		X = f['LFS'][...,self.start_idx[index]:self.stop_idx[index]]
		f.close()
		#label for clear(=0) or disrupted(=1, or weighted)
		y = np.zeros((X.shape[-1],),dtype=int)
		if self.disrupted[index]:
			#TODO: class weighting beyond constant
			y[self.disrupt_idx[index]:] = 1

		return X,y


def data_generator(dataset,batch_size,distributed=False,num_workers=0):
	"""Generate the loader objects for the train,validate, and test sets
	dataset: EceiDataset object
	distributed: (True/False) using distributed workers
	num_workers: Number of processes"""

	if not hasattr(dataset,'train_inds'):
		dataset.train_val_test_split()

	#create data subsets, based on indices created
	train_dataset = data.Subset(dataset,dataset.train_inds)
	val_dataset = data.Subset(dataset,dataset.val_inds)
	test_dataset = data.Subset(dataset,dataset.test_inds)

	#shuffle dataset each epoch for training data using DistrbutedSampler. Also splits among workers. 
	if distributed:
		train_sampler = data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None
	
	#create data loaders for train/val/test datasets
	train_loader = data.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
		num_workers=num_workers, pin_memory=True, sampler=train_sampler)

	val_loader = data.DataLoader(
		val_dataset, batch_size=batch_size, shuffle=False,
		num_workers=num_workers, pin_memory=True)

	test_loader = data.DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False,
		num_workers=num_workers, pin_memory=True)
	
	return train_loader,val_loader,test_loader
