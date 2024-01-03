#!/usr/bin python

import h5py
import numpy as np
from disruptcnn.loader import EceiDataset
import torch


root = '../ecei_d3d/'
data_root = root+'data/'
clear_file = root + 'd3d_clear_ecei.final.txt'
disrupt_file = root + 'd3d_disrupt_ecei.final.txt'
dataset = EceiDataset(data_root,clear_file,disrupt_file,flattop_only=False)

#first, form the offsets. All of this data starts at -50ms, so just take in the
#first 40,000 points (-50ms to -10ms) to form offset.
offsets = np.zeros((20,8,len(dataset)),dtype='float32')
print('Forming offsets')
for i in range(len(dataset)):
    print(i)
    shot = dataset.shot[i]
    if dataset.disrupted[i]:
        shot_type='disrupt'
    else:
        shot_type='clear'

    f = h5py.File(root+'data/'+shot_type+'/'+str(shot)+'.h5','r+')
    data = f['LFS'][...,0:40000]
    offsets[...,i] = np.mean(data,axis=-1)
    del f['offsets']
    #if not 'offsets' in f.keys():
    dset = f.create_dataset('offsets',data=offsets[...,i])
    f.close()
offsetsMean = np.mean(offsets,axis=(0,1))
offsetsStd = np.std(offsets,axis=(0,1))
np.savez('offsets.npz',offsets=offsets,offsetsMean=offsetsMean,offsetsStd=offsetsStd)
