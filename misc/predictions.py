# coding: utf-8
import torch
import sys
sys.path.append('./disruptcnn/')
from disruptcnn.main import create_model
from disruptcnn.loader import EceiDataset
from collections import OrderedDict
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time

import argparse

parser = argparse.ArgumentParser(description='Sequence Modeling - disruption ECEi')
parser.add_argument('--model-file', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--splits-file', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

root = '/gpfs/alpine/proj-shared/fus131/ecei_d3d/'
data_root = root+'data/'
clear_file = root + 'd3d_clear_ecei.final.txt'
disrupt_file = root + 'd3d_disrupt_ecei.final.txt'

def calc_prediction_single(shot_example,model,args,dataset,disrupt=False):
    shot_type = 'clear'
    if disrupt: shot_type='disrupt'

    fileTest = dataset.root+shot_type+'/'+str(shot_example)+'.h5'
    f = h5py.File(fileTest,'r')
    offset = f['offsets'][...]
    data = f['LFS'][:,:,::args.data_step] - offset[...,np.newaxis]
    f.close()

    N = data.shape[-1]
    Nseq = int(np.ceil(N/args.nsub))

    #t = -50 + np.arange(data.shape[-1])*(1e-3*args.data_step)
    #normalize

    ind_shot = np.where(dataset.shot==shot_example)[0]
    inds_shot_idxis = np.where(dataset.shot_idxi==ind_shot)[0]
    istart = int(dataset.start_idxi[inds_shot_idxis].min()/args.data_step)
    iend = int(dataset.stop_idxi[inds_shot_idxis].max()/args.data_step)
    i1s = [istart]
    i2s = [istart + args.nsub]
    while i2s[-1]<iend:
        i1s.append(i2s[-1]  - args.nrecept + 1)
        i2s.append(i1s[-1] + args.nsub)
    out = np.zeros((1,N))


    x = (data - dataset.normalize_mean[...,np.newaxis])/dataset.normalize_std[...,np.newaxis]
    x = torch.tensor(x[np.newaxis,...])
    x = x.view(1, args.input_channels, -1)
    with torch.no_grad():
        x = x.cuda()
    #    out = np.zeros((1,N))
    #    for i in range(Nseq):
    #        i1 = i*args.nsub - i*args.nrecept + i
    #        i2 = i1 + args.nsub
    #        out[...,i1+args.nrecept:i2] = model(x[...,i1:i2])[...,args.nrecept:].cpu().detach().numpy()
        for i1,i2 in zip(i1s,i2s):
            out[...,i1+args.nrecept:i2] = model(x[...,i1:i2])[...,args.nrecept:].cpu().detach().numpy()
    return out

def calc_predictions(model_file,splits_file):
    #load model file
    out = torch.load(model_file)
    
    #read and load model
    threshold = out['threshold']
    args = out['args']
    model = create_model(args)
    saved_state_dict = out['state_dict']
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda() #assumed on GPU
    model.eval()


    #create dataset object for normalizations
    dataset = EceiDataset(data_root,clear_file,disrupt_file,
                          label_balance=args.label_balance,
                          normalize=(not args.no_normalize),
                          data_step=args.data_step,
                          nsub=args.nsub,nrecept=args.nrecept,
                          flattop_only=args.flattop_only)
    
    #read validation set
    fsplits = np.load(splits_file)
    shot = fsplits['shot']
    shot_idxi = fsplits['shot_idxi']
    train_inds = fsplits['train_inds']
    val_inds = fsplits['val_inds']
    test_inds = fsplits['test_inds']
            

    #recreate the loaders with the splits from before
    #dataset.train_val_test_split(train_inds=train_inds,val_inds=val_inds,test_inds=test_inds)
    #
    #train_loader, val_loader, test_loader = data_generator(dataset, args.batch_size, 
    #                                            distributed=args.distributed,
    #                                            num_workers=args.workers,
    #                                            undersample=None, oversample=args.oversample)
    #create the indices for train/val/test split
    #calculate inference predictions for disruptive (pos) and clear (neg) sequences
    slurm_id = ''.join(filter(str.isdigit, model_file))
    fw = h5py.File('predictions_'+slurm_id+'.h5','w')
    fw.attrs['model_file'] = model_file
    fw.attrs['splits_file'] = splits_file
    for setname,inds in zip(['val','test'],[val_inds,test_inds]):
        preds = []; start_idx = []; stop_idx = []; disrupt_idx = []; disrupted = []
        grp = fw.create_group('preds_'+setname)
        shots_set = np.unique(shot[shot_idxi[inds]])
        shots_set.sort()
        tstart = time.time()
        for i,shoti in enumerate(shots_set): 
            ind = np.where(dataset.shot==shoti)[0]
            start_idx += [dataset.start_idx[ind]]
            stop_idx += [dataset.stop_idx[ind]]
            disrupt_idx += [dataset.disrupt_idx[ind]]
            disrupted += [dataset.disrupted[ind]]
            pred = calc_prediction_single(shoti,model,args,dataset,disrupt=disrupted[i])
            dset = grp.create_dataset(str(shoti),shape=pred.shape, data=pred)
            print("Index %d/%d, Shot %d completed, Time: %0.2f" % (i,shotsi_set.size,shoti,time.time()-tstart))
        fw.create_dataset('shots_'+setname,shape=shots_set.shape,data=shots_set)
        fw.create_dataset('start_idx_'+setname,shape=(len(start_idx),),data=np.array(start_idx))
        fw.create_dataset('stop_idx_'+setname,shape=(len(stop_idx),),data=np.array(stop_idx))
        fw.create_dataset('disrupt_idx_'_setname,shape=(len(disrupt_idx),),data=np.array(disrupt_idx))
        fw.create_dataset('disrupted_'+setname,shape=(len(disrupted),),data=np.array(disrupted))
        fw.close()
        
#    np.savez('validation_predictions_'+slurm_id+'.npz',preds=preds, shots_val=shots_val,
#                                                       model_file=model_file, splits_file=splits_file,
#                                                       start_idx_val=start_idx_val,
#                                                       stop_idx_val=stop_idx_val,
#                                                       disrupt_idx_val=disrupt_idx_val,
#                                                       disrupted_val=disrupted_val)

if __name__ == "__main__":
    args = parser.parse_args()
    tstart = time.time()
    calc_predictions(args.model_file,args.splits_file)
    print('Time Elapsed: %0.2f' % (time.time()-tstart))
