#!/usr/bin python

import numpy as np
import torch.utils.data as data
import h5py
from sklearn.model_selection import train_test_split
import os
#from torch.utils.data import SubsetRandomSampler
from disruptcnn.sampler import StratifiedSampler

class EceiDataset(data.Dataset):
    """ECEi dataset"""

    def __init__(self,root,clear_file,disrupt_file,
                      train=True,flattop_only=True,
                      Twarn=300,
                      test=0,test_indices=None,
                      label_balance='const',normalize=True,data_step=1,
                      nsub=None,nrecept=None):
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
        test: training set size (# shots), for testing of overfitting purposes
        test_indices: List of specific global indices (len(test_indices) must match test)
        label_balance: For imbalanced label sets, uses weight to balance in binary cross entropy  (default True)
        data_step: step to take in indexing the data
        nsub: Subsequence length to use
        nrecept: Receptive field length of model
        """
        self.root = root
        self.train = train #training set or test set TODO: Do I need this?

        self.Twarn = Twarn #in ms
        self.test = test
        self.label_balance = label_balance
        self.normalize = normalize
        self.data_step = data_step
        self.nsub = nsub
        self.nrecept = nrecept

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

        #start_idx < idx < disrupt_idx: non-disruptive
        #disrupt_idx < idx < stop_idx: disruptive
        self.disrupt_idx = np.ceil((tdisrupt-self.Twarn-tstarts)/dt).astype(int)
        self.disrupt_idx[tdisrupt<0] = -1000 #TODO: should this be something else? nan isnt possible with int
        self.disrupted = self.disrupt_idx>0 #True if disrupted, False if not
        
        self.zero_idx = np.ceil((0.-tstarts)/dt).astype(int)
        if flattop_only:
            self.start_idx = np.ceil((tflatstarts-tstarts)/dt).astype(int)
            tend = np.maximum(tdisrupt,np.minimum(tstops,tflatstops)) #minimum ensures tflatstops<tstops
            #tend = np.maximum(tdisrupt,tflatstops)
        else:
            self.start_idx = np.ceil((0.-tstarts)/dt).astype(int)
            tend = np.maximum(tdisrupt,tstops)
        self.stop_idx = np.floor((tend-tstarts)/dt).astype(int)

        self.shot = data_all[:,0].astype(int)
        self.length = len(self.shot)

        #create offsets placeholder
        filename = self.create_filename(0)
        f = h5py.File(filename,'r')
        self.offsets = np.zeros(f['offsets'].shape+(self.shot.size,),dtype=f['offsets'].dtype)
        f.close()

        #read in normalization data (per channel)
        f = np.load(self.root+'normalization.npz')
        if flattop_only:
            self.normalize_mean = f['mean_flat']
            self.normalize_std = f['std_flat']
        else:
            self.normalize_mean = f['mean_all']
            self.normalize_std = f['std_all']

        #split shots into subsequences (allows uniform subsequence length, easier batching)
        self.shots2seqs()

        #create label weights
        #TODO: Is this needed here? Called in train_val_split
        self.calc_label_weights()

        #testing setup
        if self.test > 0:
            labels = self.disruptedi
            self.length = self.test
            if self.test==1:
                disinds = np.where(self.disruptedi)[0]
                self.test_indices = disinds[np.random.randint(disinds.size,size=1)]
            else:
                if test_indices is None:
                    disinds = np.where(self.disruptedi)[0]
                    disinds = np.random.choice(disinds,size=int(self.test/2))
                    nondisinds = np.where(self.disruptedi==0)[0]
                    nondisinds = np.random.choice(nondisinds,size=self.test - int(self.test/2))
                    self.test_indices = np.concatenate([disinds,nondisinds])
                else:
                    assert len(test_indices)==self.test
                    self.test_indices = np.array(test_indices)
            #if self.test<32:
            #    #preload test data for speed
            #    self.test_data = []
            #    for ind in self.test_indices:
            #        self.test_data += [self.read_data(ind)]

    def shots2seqs(self):
        self.shot_idxi = []; self.start_idxi = []; self.stop_idxi = []; self.disrupt_idxi = []
        for s in range(len(self.shot)):
            N = int((self.stop_idx[s] - self.start_idx[s] + 1)/self.data_step) #length of entire sequence
            num_seq_frac = (N - self.nsub)/float(self.nsub - self.nrecept + 1)+1
            num_seq = np.ceil(num_seq_frac).astype(int)
            if num_seq<1: num_seq = 1 #try to force at least 1 subsequence from shot
            #determine if there is enough additional sequence points to cover the 
            #proposed number of sequences
            Nseq = self.nsub + (num_seq - 1)*(self.nsub - self.nrecept + 1)
            if ((self.start_idx[s]>self.zero_idx[s]) & 
               ((self.start_idx[s] - self.zero_idx[s] + 1) > (Nseq - N)*self.data_step)):
                self.start_idx[s] -= (Nseq - N)*self.data_step
            else:
                num_seq -= 1
                Nseq = self.nsub + (num_seq - 1)*(self.nsub - self.nrecept + 1)
                self.start_idx[s] += (N - Nseq)*self.data_step
            
            for m in range(num_seq):
                self.shot_idxi += [s]
                self.start_idxi += [self.start_idx[s] + (m*self.nsub - m*self.nrecept + m)*self.data_step]
                self.stop_idxi  += [self.start_idx[s] + ((m+1)*self.nsub - m*self.nrecept + m)*self.data_step]
                if ((self.start_idxi[-1]<=self.disrupt_idx[s]) & (self.stop_idxi[-1]>=self.disrupt_idx[s])):
                    self.disrupt_idxi += [self.disrupt_idx[s]]
                else:
                    self.disrupt_idxi += [-1000]
        
        self.shot_idxi = np.array(self.shot_idxi); self.start_idxi = np.array(self.start_idxi); 
        self.stop_idxi = np.array(self.stop_idxi); self.disrupt_idxi = np.array(self.disrupt_idxi)
        self.disruptedi = self.disrupt_idxi>0
        self.length = len(self.shot_idxi)

    def calc_label_weights(self,inds=None):
        """Calculated weights to use in the criterion"""
        #for now, do a constant weight on the disrupted class, to balance the unbalanced set
        #TODO implement increasing weight towards final disruption
        if inds is None: inds = np.arange(len(self.shot))
        if 'const' in self.label_balance:
            N = np.sum(self.stop_idxi[inds] - self.start_idxi[inds])
            disinds = inds[self.disruptedi[inds]]
            Ndisrupt = np.sum(self.stop_idxi[disinds] - self.disrupt_idxi[disinds])
            Nnondisrupt = N - Ndisrupt
            self.pos_weight = 0.5*N/Ndisrupt
            self.neg_weight = 0.5*N/Nnondisrupt
        else:
            self.pos_weight = 1
            self.neg_weight = 1

    def train_val_test_split(self,sizes=[0.8,0.1,0.1],random_seed=42,
                                  train_inds=None,val_inds=None,test_inds=None):
        """Creates indices to split data into train, validation, and test datasets. 
        Stratifies to ensure each group has class structure consistent with original class balance
        sizes: Fractional size of train, validation, and test data sizes (3 element array or list)
        """
        assert(np.array(sizes).size==3)
        assert(np.isclose(np.sum(sizes),1.0))

        #TODO: make labels based on each point, NOT just whether has disruptive points?
        labels = self.disrupted

        if self.test > 0:
            self.train_inds = self.test_indices
            self.val_inds = []
            self.test_inds = []
        if train_inds is not None:
            self.train_inds = train_inds
            self.val_inds = val_inds
            self.test_inds = test_inds
        else:
            #split first according to shot classification
            train_shot_inds,valtest_shot_inds,train_labels,valtest_labels = train_test_split(np.arange(len(self.shot)),labels,
                                                                                        stratify=labels,
                                                                                        test_size=np.sum(sizes[1:]),
                                                                                        random_state=random_seed)
            val_shot_inds, test_shot_inds, _, _ = train_test_split(valtest_shot_inds,valtest_labels,
                                                                   stratify=valtest_labels,
                                                                   test_size=sizes[2]/np.sum(sizes[1:]),
                                                                   random_state=random_seed)
            #now get the subsequnces which belong to the shot classification split
            # (this makes sure no bleeding of shots between train/val/test splits)
            self.train_inds = np.where(np.in1d(self.shot_idxi,train_shot_inds))[0]
            self.val_inds = np.where(np.in1d(self.shot_idxi,val_shot_inds))[0]
            self.test_inds = np.where(np.in1d(self.shot_idxi,test_shot_inds))[0]
        self.calc_label_weights(inds=self.train_inds)

    def create_filename(self,index):
        if self.disrupted[index]:
            shot_type='disrupt'
        else:
            shot_type='clear'

        return self.root+shot_type+'/'+str(self.shot[index])+'.h5'

    def read_data(self,index):
        shot_index = self.shot_idxi[index]
        filename = self.create_filename(shot_index)
        f = h5py.File(filename,'r')
        #check if weve read in offsets yet
        if np.all(self.offsets[...,shot_index]==0):
            self.offsets[...,shot_index] = f['offsets'][...]
        #read data, remove offset
        X = f['LFS'][...,self.start_idxi[index]:self.stop_idxi[index]][...,::self.data_step] - self.offsets[...,shot_index][...,np.newaxis]
        if self.normalize:
            X = (X - self.normalize_mean[...,np.newaxis])/self.normalize_std[...,np.newaxis]
        f.close()
        return X


    def __len__(self):
        """Return total number of sequences"""
        return self.length

    def __getitem__(self,index):
        """Read the data from file. Reads the entire sequence from the shot file"""
        if (self.test > 0) & (hasattr(self,'test_data')):
            ind_test = np.where(self.test_indices==index)[0][0] #since the loader has inds up to len(self.shot)
            X = self.test_data[ind_test]
        else:
            X = self.read_data(index)

        #label for clear(=0) or disrupted(=1, or weighted)
        target = np.zeros((X.shape[-1]),dtype=X.dtype)
        weight = self.neg_weight*np.ones((X.shape[-1]),dtype=X.dtype)
        if self.disruptedi[index]:
            #TODO: class weighting beyond constant
            target[int((self.disrupt_idxi[index]-self.start_idxi[index]+1)/self.data_step):] = 1
            weight[int((self.disrupt_idxi[index]-self.start_idxi[index]+1)/self.data_step):] = self.pos_weight

        return X,target,index.item(),weight



def data_generator(dataset,batch_size,distributed=False,num_workers=0,num_replicas=None,rank=None,undersample=None):
    """Generate the loader objects for the train,validate, and test sets
    dataset: EceiDataset object
    distributed (optional): (True/False) using distributed workers
    num_workers (optional): Number of processes
    undersample (optional): Fraction of neg/pos samples 
    """

    if not hasattr(dataset,'train_inds'):
        dataset.train_val_test_split()

    #create data subsets, based on indices created
    train_dataset = data.Subset(dataset,dataset.train_inds)
    val_dataset = data.Subset(dataset,dataset.val_inds)
    test_dataset = data.Subset(dataset,dataset.test_inds)

    #shuffle dataset each epoch for training data using DistrbutedSampler. Also splits among workers. 
    train_sampler = StratifiedSampler(train_dataset,num_replicas=num_replicas,rank=rank,stratify=dataset.disruptedi[dataset.train_inds],distributed=distributed,undersample=undersample)
    val_sampler = StratifiedSampler(val_dataset,num_replicas=num_replicas,rank=rank,stratify=dataset.disruptedi[dataset.val_inds],distributed=distributed,undersample=undersample)
   
    #redo class weights, since they are based on non-undersampled datasets
    if undersample is not None:
        inds = np.array([dataset.train_inds[i] for i in train_sampler])
        dataset.calc_label_weights(inds=inds)

    #create data loaders for train/val/test datasets
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)#,
        #drop_last=True)

    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=val_sampler)#,
        #drop_last=True)
        #val_dataset, batch_size=batch_size, shuffle=False,
        #num_workers=num_workers, pin_memory=True)

    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        drop_last=True)
    
    return train_loader,val_loader,test_loader
