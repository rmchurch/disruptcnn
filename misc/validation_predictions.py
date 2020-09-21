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
    #t = -50 + np.arange(data.shape[-1])*(1e-3*args.data_step)
    #normalize
    x = (data - dataset.normalize_mean[...,np.newaxis])/dataset.normalize_std[...,np.newaxis]
    x = torch.tensor(x[np.newaxis,...])
    x = x.view(1, args.input_channels, -1)
    x = x.cuda()
    return model(x).cpu().detach().numpy()

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

    #read validation set
    fsplits = np.load(splits_file)
    shot = fsplits['shot']
    shot_idxi = fsplits['shot_idxi']
    val_inds = fsplits['val_inds']
    
    shots_val = np.unique(shot[shot_idxi[val_inds]])
    shots_val.sort()

    #create dataset object for normalizations
    dataset = EceiDataset(data_root,clear_file,disrupt_file,
                          label_balance=args.label_balance,
                          normalize=(not args.no_normalize),
                          data_step=args.data_step,
                          nsub=args.nsub,nrecept=args.nrecept,
                          flattop_only=args.flattop_only)
    #create the indices for train/val/test split
    #calculate inference predictions for disruptive (pos) and clear (neg) sequences
    preds = []; start_idx_val = []; stop_idx_val = []; disrupt_idx_val = []; disrupted_val = []
    slurm_id = ''.join(filter(str.isdigit, model_file))
    fw = h5py.File('validation_predictions_'+slurm_id+'.h5','w')
    fw.attrs['model_file'] = model_file
    fw.attrs['splits_file'] = splits_file
    grp = fw.create_group('preds')
    for i,shot in enumerate(shots_val): 
        ind = np.where(dataset.shot==shot)[0]
        start_idx_val += [dataset.start_idx[ind]]
        stop_idx_val += [dataset.stop_idx[ind]]
        disrupt_idx_val += [dataset.disrupt_idx[ind]]
        disrupted_val += [dataset.disrupted[ind]]
        pred = calc_prediction_single(shot,model,args,dataset,disrupt=disrupted_val[i])
        dset = grp.create_dataset(str(shot),shape=pred.shape, data=pred)
    fw.create_dataset('shots_val',shape=shots_val.shape,data=shots_val)
    fw.create_dataset('start_idx_val',shape=(len(start_idx_val),),data=np.array(start_idx_val))
    fw.create_dataset('stop_idx_val',shape=(len(stop_idx_val),),data=np.array(stop_idx_val))
    fw.create_dataset('disrupt_idx_val',shape=(len(disrupt_idx_val),),data=np.array(disrupt_idx_val))
    fw.create_dataset('disrupted_val',shape=(len(disrupted_val),),data=np.array(disrupted_val))
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
