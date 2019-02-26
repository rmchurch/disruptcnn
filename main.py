#!/usr/bin python
import random
import warnings
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from disruptcnn.loader import data_generator, EceiDataset
from disruptcnn.model import TCN
import time
from tensorboardX import SummaryWriter
import os, psutil, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Sequence Modeling - disruption ECEi')
#model specific
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
parser.add_argument('--input-channels', type=int, default=160,
                    help='number of ECEi channels (should generalize to read data) (default: 160)')
parser.add_argument('--n-classes', type=int, default=1,
                    help='number classification classes (1 for binary classification) (default: 1)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--kernel-size', type=int, default=15,
                    help='kernel size (default: 15)')
parser.add_argument('--dilation-size', type=int, default=10,
                    help='kernel size (default: 10)')
parser.add_argument('--levels', type=int, default=5,
                    help='# of levels (default: 5)')
parser.add_argument('--nhid', type=int, default=20,
                    help='number of hidden units per layer (default: 20)')
parser.add_argument('--nrecept', type=int, default=300000,
                    help='receptive field sequence length (default: 300000)')
parser.add_argument('--nsub', type=int, default=5000000,
                    help='sequence length to optimize over, usually '
                    'set by GPU memory contrainsts(default: 5000000)')
#learning specific
parser.add_argument('--multiplier-warmup', type=float, default=8,
                    help='warmup divide initial lr factor (default: 8)')
parser.add_argument('--iterations-warmup', type=int, default=200,
                    help='LR warmup iterations (default: 200)')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--iterations-valid', type=int, default=200,
                    help='iteration period to run validation(default: 200)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--weight-balance', action='store_false',
                    help='Balance an imbalanced dataset with weight in cross-entropy loss (default: True)')
#other
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')
parser.add_argument('--log-interval', type=int, default=1,
                    help='Frequency of logging (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--test', default=0, type=int, metavar='N',
                    help='runs on single example, to verify model can overfit (default: 0)')
parser.add_argument('--test-indices', default=None, nargs='*',type=int,
                    help='list of global indices to use (default: None)')



#TODO Generalize
#batch_size = 1#args.batch_size
#n_classes = 1 #for binary classification
#input_channels = 160
#steps = 0
#seq_length = int(Nseq) #TODO: This might be different from how I defined it in loader.py
#dilation_sizes = [1,10,100,1000,6783] #dilation=10, except last which is set to give receptive field ~Nmodel=300,000
#Nsub = 5000000 #found by taking receptive field, and scaling for 15GB of GPU memory #TODO automate
#Nrecept = 300000
root = '/scratch/gpfs/rmc2/ecei_d3d/'
data_root = root+'data/'
clear_file = root + 'd3d_clear_ecei.final.txt'
disrupt_file = root + 'd3d_disrupt_ecei.final.txt'



def main():
    args = parser.parse_args()

    assert (args.batch_size==1), "Currently need batch_size=1, due to variable length sequences"
    assert torch.cuda.is_available(), "GPU is currently required"

    args.world_size = int(os.environ['SLURM_NTASKS'])
    args.rank = int(os.environ['SLURM_PROCID'])
    args.tstart = tstart

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        #TODO Generalize for non-GPU? This requires GPU
        args.gpu = int(os.environ['SLURM_LOCALID'])
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu,ngpus_per_node,args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    #TODO handle distributed case for tensorboard logging
    is_writer =  not args.distributed or (args.distributed and args.rank == 0)
    if is_writer:
        writer = SummaryWriter()
        #save args
        for argname in vars(args):
            writer.add_text(argname,str(getattr(args,argname)))

    #create TCN model
    model = create_model(args)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            #TODO: I am making batch_size per process. Generalize?
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            #args.batch_size = int(args.batch_size / ngpus_per_node)
            #args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        # DataParallel will divide and allocate batch_size to all available GPUs
        # model = torch.nn.DataParallel(model).cuda()

    print(args)
    dataset = EceiDataset(data_root,clear_file,disrupt_file,
                          test=args.test,test_indices=args.test_indices,
                          weight_balance=args.weight_balance)
    #create the indices for train/val/test split
    dataset.train_val_test_split()
    #create data loaders
    train_loader, val_loader, test_loader = data_generator(dataset, args.batch_size, 
                                                            distributed=args.distributed,
                                                            num_workers=args.workers)

    #TODO generalize momentum?
    #TODO implement general optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    #gradual linear increasing learning rate for warmup
    lambda1=lambda iteration: (1.-1./args.multiplier_warmup)/args.iterations_warmup*iteration+1./args.multiplier_warmup
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)
    #decaying learning rate scheduler for after warmup
    #TODO generalize factor?
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5)

    steps = 0
    total_loss = 0
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, (data, target, global_index, weight) in enumerate(train_loader):
            model.train()
            iteration = epoch*len(train_loader) + batch_idx

            #learning rate scheduler
            if iteration < args.iterations_warmup:
                scheduler_warmup.step(iteration)
            else:
                #scheduler_plateau.step(train_loss)
                #TODO change to be general outside of test
                if iteration % args.test == 0:
                    scheduler_plateau.step(total_loss)
                    total_loss = 0

            #train single iteration
            train_loss = train_seq(data,target,weight,model,optimizer,args)
            if is_writer: writer.add_scalar('train_loss',train_loss,iteration)
            steps += data.shape[0]*data.shape[-1]
            total_loss += train_loss

            #validate
            if (iteration>0) & (iteration % args.iterations_valid == 0) & (args.test==0):
                valid_loss = evaluate(val_loader, model, args)
                #TODO Replace when accuracy written
                acc = valid_loss
                 
                writer.add_scalar('valid_loss',valid_loss,iteration)
                # remember best acc and save checkpoint
                is_best = acc > best_acc
                best_acc = max(acc, best_acc)
                 
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best,filename='checkpoint.'+os.environ['SLURM_JOB_ID']+'.pth.tar')
            
            
            if batch_idx % args.log_interval == 0:
                lr_epoch = [ group['lr'] for group in optimizer.param_groups ][0]
                print('Train Epoch: %d [%d/%d (%0.2f%%)]\tIndex: %d\tDisrupted: %d\tLoss: %0.6e\tSteps: %d\tTime: %0.2f\tMem: %0.1f\tLR: %0.2e' % (
                            epoch, (batch_idx*args.world_size+args.rank), len(train_loader.dataset),
                            100. * (batch_idx*args.world_size+args.rank)  / len(train_loader.dataset), global_index, train_loader.dataset.dataset.disrupted[global_index],train_loss, steps,(time.time()-args.tstart),psutil.virtual_memory().used/1024**3.,lr_epoch))
        
        print('Train Epoch: %d \tTotal Loss: %0.6e\tSteps: %d\tTime: %0.2f\tLR: %0.2e' % (
                            epoch, total_loss, steps,(time.time()-args.tstart),lr_epoch))
    if (args.test>0):
        plt.figure(figsize=[6.40,7.40])
        for batch_idx, (data, target, global_index, weight) in enumerate(train_loader):
            with torch.no_grad():
                data, target, weight = data.cuda(non_blocking=True), target.cuda(non_blocking=True), weight.cuda(non_blocking=True)
                data = data.view(args.batch_size, args.input_channels, -1)
                output = model(data)
                loss = F.binary_cross_entropy(output[...,args.nrecept-1:], target[...,args.nrecept-1:], weight=weight[...,args.nrecept-1:],reduction='sum').item()/(output.shape[-1]-args.nrecept+1)
                train_loss = process_seq(data,target,args.nsub,args.nrecept,model,
                                          optimizer=None,weight=weight,
                                          train=False,clip=args.clip)
                plt.clf()
                plt.subplot(311)
                plt.plot(data[0,0,:].detach().cpu().numpy())
                plt.title('Loss: %0.4e' % loss)
                plt.subplot(312)
                plt.plot(output[...,args.nrecept-1:].detach().cpu().numpy()[0,:])
                plt.plot(target[...,args.nrecept-1:].detach().cpu().numpy()[0,:],'--')
                plt.subplot(313)
                plt.plot(weight[...,args.nrecept-1:].detach().cpu().numpy()[0,:])
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.05)
                plt.savefig('test_output_'+str(int(os.environ['SLURM_JOB_ID']))+'_ind_'+str(global_index.item())+'.png')

    if is_writer: writer.close()
    time.sleep(180) #allow all processes to finish

def train_seq(data, target, weight, model, optimizer, args):
    '''Takes a batch sequence and trains, splitting if needed'''
    if args.cuda: 
        data, target, weight  = data.cuda(non_blocking=True), \
                                target.cuda(non_blocking=True), \
                                weight.cuda(non_blocking=True)
    data = data.view(args.batch_size, args.input_channels, -1)
    #split data into subsequences to process
    train_loss = process_seq(data,target,args.nsub,args.nrecept,model,
                              optimizer=optimizer,weight=weight,
                              train=True,clip=args.clip)
    return train_loss


def process_seq(data,target,Nsub,Nrecept,model,optimizer=None,train=True,weight=None,clip=None):
    '''Splits apart sequence into equal, overlapping subsequences of length Nsub, with overlap Nrecept
    Does accumulated gradients method to avoid large GPU memory usage
    '''
    if weight is None: weight = torch.ones(target.shape).cuda()
    N = data.shape[-1] #length of entire sequence
    num_seq_frac = (N - Nsub)/float(Nsub - Nrecept + 1)+1
    num_seq = np.ceil(num_seq_frac).astype(int)
    total_losses = 0
    for m in range(num_seq):
        start_idx =    m*Nsub - m*Nrecept + m
        stop_idx = (m+1)*Nsub - m*Nrecept + m
        if stop_idx>N: stop_idx = N
        if ((stop_idx-start_idx)<Nrecept):
            start_idx = stop_idx - Nrecept
        #reverse to ensure disruptive portion never split
        #TODO should I instead just split in half the sequence?
        tmp = start_idx.copy()
        start_idx = N - stop_idx
        stop_idx = N - tmp

        if optimizer is not None:
            optimizer.zero_grad()
        ys = model(data[...,start_idx:stop_idx])
        ts = target[...,start_idx:stop_idx]
        ws = weight[...,start_idx:stop_idx]
        #do mean of loss by hand to handle unequal sequence lengths
        loss = F.binary_cross_entropy(ys[...,Nrecept-1:],ts[...,Nrecept-1:],weight=ws[...,Nrecept-1:],reduction='sum')/(N-Nrecept+1)
        if train: loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        total_losses += loss.item()
        if optimizer is not None:
            optimizer.step()
    return total_losses


#for the validation and test set
def evaluate(val_loader,model,args):
    model.eval()
    loss = 0
    correct = 0
    first = True
    with torch.no_grad(): #turns off backprop, saves computation
        for data, target,_,weight in val_loader:
            data, target, weight = data.cuda(non_blocking=True), target.cuda(non_blocking=True), weight.cuda(non_blocking=True)
            data = data.view(args.batch_size, args.input_channels, -1)
            output = model(data)
            #loss += F.binary_cross_entropy(output, target, size_average=False).item()
            loss += F.binary_cross_entropy(output[...,args.nrecept-1:], target[...,args.nrecept-1:], weight=weight[...,args.nrecept-1:],reduction='sum').item()/(output.shape[-1]-args.nrecept+1)
            if ((val_loader.dataset.dataset.disrupted[global_index]==1)):
                plt.clf()
                plt.plot(output[...,Nrecept-1:].detach().cpu().numpy()[0,:])
                plt.plot(target[...,Nrecept-1:].detach().cpu().numpy()[0,:])
                plt.savefig('output_rank_'+str(args.rank)+'_ind_'+str(int(global_index))+'.png')
                first = False
            #TODO: Enter validation loss here, modify from this simple
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            correct = -1
            print('Validation  [%d/%d (%0.2f%%)]\tIndex: %d\tDisrupted: %d\tLoss: %0.6e\tTime: %0.2f\tMem: %0.1f' % (
                         (batch_idx*args.world_size+args.rank), len(val_loader.dataset),
                        100. * (batch_idx*args.world_size+args.rank)  / len(val_loader.dataset), global_index, val_loader.dataset.dataset.disrupted[global_index], loss, (time.time()-args.tstart),psutil.virtual_memory().used/1024**3.))


        loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return loss


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.'+os.environ['SLURM_JOB_ID']+'.pth.tar')


def create_model(args):
    channel_sizes = [args.nhid] * args.levels
    #first, verify that the requested args.nrecept and args.dilation_size are sufficient for the 
    #args.nrecept requested
    nrecepttotal = calc_seq_length(args.kernel_size,args.dilation_size,args.levels)
    assert nrecepttotal >= args.nrecept
    #second, adjust last level dilation factor to put receptive field close to requested args.nrecept
    nlastlevel = calc_seq_length(args.kernel_size,args.dilation_size,args.levels-1)
    last_dilation = int(np.ceil((args.nrecept - nlastlevel)/(2.*(args.kernel_size-1))))
    dilation_sizes = (args.dilation_size**np.arange(args.levels-1)).tolist() + [last_dilation]
    #reset args.nrecept with the actual receptive field
    args.nrecept = calc_seq_length(args.kernel_size,dilation_sizes,args.levels)

    model = TCN(args.input_channels, args.n_classes, channel_sizes, 
                kernel_size=args.kernel_size, 
                dropout=args.dropout,
                dilation_size=dilation_sizes)
    return model

def calc_seq_length(kernel_size,dilation_sizes,nlevel):
    """Assumes kernel_size scalar, dilation_size exponential increase"""
    if np.isscalar(dilation_sizes): dilation_sizes = dilation_sizes**np.arange(nlevel)
    return 1 + 2*(kernel_size-1)*np.sum(dilation_sizes)


if __name__ == "__main__":
    tstart = time.time()
    main()
    #TODO: test data set, create final statistics (ROC? Printed recall/precision?)
