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
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--iterations-valid', type=int, const=200, nargs='?',
                    help='iteration period to run validation(default: 1 epoch if no flag, 200 iterations if flag but no value)')
parser.add_argument('--iterations-warmup', type=int, const=200, nargs='?',
                    help='LR warmup iterations (default: 5 epochs if no flag, 200 iterations if flag but no value)')
parser.add_argument('--multiplier-warmup', type=float, default=8,
                    help='warmup divide initial lr factor (default: 8)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--label-balance', type=str,default='const',
                    help="Type of label balancing. 'const' or 'none', (default: const)")
parser.add_argument('--accumulate', action='store_true',
                    help='accumulate gradients over entire batch, i.e. shot (default: False)')
parser.add_argument('--undersample', type=float, nargs='?',const=1.0,
                    help='fraction of non-disruptive/disruptive subsequences (default: None if no flag, 1.0 if flag but no value)')
#other
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')
parser.add_argument('--log-interval', type=int, const=100, nargs='?',
                    help='Frequency of logging (default: iterations-valid or test if no flag, 100 iterations if flag but no value)')
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-step', default=1, type=int,
                    help='step to take in indexing the data')
parser.add_argument('--test', default=0, type=int, metavar='N',
                    help='runs on single example, to verify model can overfit (default: 0)')
parser.add_argument('--test-indices', default=None, nargs='*',type=int,
                    help='list of global indices to use (default: None)')
parser.add_argument('--no-normalize', action='store_true',
                    help='dont normalize the data (default: False)')
parser.add_argument('--lr-finder', action='store_true',
                    help='Learning rate finder test (default: False)')



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

    #assert (args.batch_size==1), "Currently need batch_size=1, due to variable length sequences"
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

    if (args.test>0) and (args.test < args.batch_size): args.batch_size = args.test

    print(args)
    dataset = EceiDataset(data_root,clear_file,disrupt_file,
                          test=args.test,test_indices=args.test_indices,
                          label_balance=args.label_balance,
                          normalize=(not args.no_normalize),
                          data_step=args.data_step,
                          nsub=args.nsub,nrecept=args.nrecept)
    #create the indices for train/val/test split
    dataset.train_val_test_split()
    #create data loaders
    train_loader, val_loader, test_loader = data_generator(dataset, args.batch_size, 
                                                            distributed=args.distributed,
                                                            num_workers=args.workers,
                                                            undersample=args.undersample)

    #set defaults for iterations_warmup (5 epochs) and iterations_valid (1 epoch)
    #TODO Add separate argsparse for epochs_warmup and epochs_valid?
    if args.iterations_warmup is None: args.iterations_warmup = 5*len(train_loader)
    if args.iterations_valid is None: args.iterations_valid = len(train_loader)
    if args.log_interval is None: 
        if args.test==0:
            args.log_interval = args.iterations_valid
        else:
            args.log_interval = len(train_loader)

    #TODO Generalize
    args.thresholds = np.linspace(0.1,0.9,9)

    #TODO generalize momentum?
    #TODO implement general optimizer
    if not args.lr_finder:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        #gradual linear increasing learning rate for warmup
        lambda1=lambda iteration: (1.-1./args.multiplier_warmup)/args.iterations_warmup*iteration+1./args.multiplier_warmup
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)
        #decaying learning rate scheduler for after warmup
        #TODO generalize factor?
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5)
    else:
        lr_history = {"lr": [], "loss": []}
        ninterval = 800 #number of intervals (one interval is one learning rate value)
        niter_per_interval = 1 #number of iterations per interval
        niter = ninterval*niter_per_interval
        args.epochs = int(np.ceil(niter/len(train_loader)))
        args.lr = 1e-5 #start lr
        lr_end = 1e-2
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        lambda1=lambda iteration: (lr_end/args.lr)**(iteration/niter)
        scheduler_lrfinder = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            #if args.gpu is not None:
            #    # best_acc may be from a checkpoint from a different GPU
            #    best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #this autotunes algo on GPU. If variable input (like before with single shot), would
    #be worse performance
    cudnn.benchmark = True


    #main training loop
    steps = 0
    total_loss = 0
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        for batch_idx, (data, target, global_index, weight) in enumerate(train_loader):
            model.train()
            iteration = epoch*len(train_loader) + batch_idx
            args.iteration = iteration

            #learning rate scheduler
            if args.lr_finder:
                if (iteration>0) and (iteration % niter_per_interval == 0):
                    scheduler_lrfinder.step()
                    lr_epoch = [ group['lr'] for group in optimizer.param_groups ][0]
                    lr_history["lr"].append(lr_epoch)
                    lr_history["loss"].append(total_loss)
                    np.savez('lr_finder_'+str(int(os.environ['SLURM_JOB_ID']))+'.npz',lr=lr_history["lr"],loss=lr_history["loss"])
                    total_loss = 0
            else:
                if iteration < args.iterations_warmup:
                    scheduler_warmup.step(iteration)
                else:
                    #TODO change to be general outside of test
                    if args.test==0:
                       if (iteration>0) and (iteration % args.iterations_valid == 0):
                            #TODO Decide if use validation loss instead
                            scheduler_plateau.step(total_loss)
                    else:
                        if (iteration>0) and (iteration % len(train_loader) == 0):
                            scheduler_plateau.step(total_loss)

            #train single iteration
            train_loss = train_seq(data,target,weight,model,optimizer,args)
            if is_writer: writer.add_scalar('train_loss',train_loss,iteration)
            steps += data.shape[0]*data.shape[-1]
            total_loss += train_loss

            #validate
            if (iteration>0) & (iteration % args.iterations_valid == 0) & (args.test==0):
                valid_loss, valid_acc, valid_f1 = evaluate(val_loader, model, args)
                acc = valid_acc
                
                if is_writer: 
                    writer.add_scalar('valid_loss',valid_loss,iteration)
                    writer.add_scalar('valid_acc',valid_acc,iteration)
                    writer.add_scalar('valid_f1',valid_f1,iteration)
                # remember best acc and save checkpoint
                is_best = acc > best_acc
                best_acc = max(acc, best_acc)
                 
                if (not args.multiprocessing_distributed and args.rank==0) or \
                   (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best,filename='checkpoint.'+os.environ['SLURM_JOB_ID']+'.pth.tar')
            
            #log training 
            if batch_idx % args.log_interval == 0:
                lr_epoch = [ group['lr'] for group in optimizer.param_groups ][0]
                print('Train Epoch: %d [%d/%d (%0.2f%%)]\tIteration: %d\tDisrupted: %0.4f\tLoss: %0.6e\tSteps: %d\tTime: %0.2f\tMem: %0.1f\tLR: %0.2e' % (
                            epoch, batch_idx, len(train_loader), 100. * (batch_idx / len(train_loader)), iteration,
                            np.sum(train_loader.dataset.dataset.disruptedi[global_index])/global_index.size(), total_loss/args.log_interval, steps,(time.time()-args.tstart),psutil.virtual_memory().used/1024**3.,lr_epoch))
                total_loss = 0
    

    print("Main training loop ended")


    if (args.test>0):
        plt.figure(figsize=[6.40,7.40])
        for batch_idx, (data, target, global_index, weight) in enumerate(train_loader):
            with torch.no_grad():
                data, target, weight = data.cuda(non_blocking=True), target.cuda(non_blocking=True), weight.cuda(non_blocking=True)
                data = data.view(data.shape[0], args.input_channels, -1)
                output = model(data)
                loss = F.binary_cross_entropy(output[...,args.nrecept-1:], target[...,args.nrecept-1:], weight=weight[...,args.nrecept-1:],reduction='sum').item()/(output.shape[-1]-args.nrecept+1)

                for (i,gi) in enumerate(global_index):
                    plot_output(data[i,...][np.newaxis,...],output[i,...][np.newaxis,...],target[i,...][np.newaxis,...],weight[i,...][np.newaxis,...],args,
                           filename='test_output_'+str(int(os.environ['SLURM_JOB_ID']))+'_ind_'+str(global_index.item())+'.png',
                           title='Loss: %0.4e' % loss)

    if args.lr_finder:
        plt.figure()
        plt.plot(lr_history["lr"],lr_history["loss"])
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim([np.array(lr_history["loss"]).min(),lr_history["loss"][0]])
        plt.savefig('lr_finder_'+str(int(os.environ['SLURM_JOB_ID']))+'.png')
        np.savez('lr_finder_'+str(int(os.environ['SLURM_JOB_ID']))+'.npz',lr=lr_history["lr"],loss=lr_history["loss"])

    if is_writer: writer.close()
    time.sleep(180) #allow all processes to finish


def plot_output(data,output,target,weight,args,filename='output.png',title=''):
    plt.clf()
    plt.subplot(311)
    plt.plot(data[0,0,:].detach().cpu().numpy())
    plt.title(title)
    plt.subplot(312)
    plt.plot(output[...,args.nrecept-1:].detach().cpu().numpy()[0,:])
    plt.plot(target[...,args.nrecept-1:].detach().cpu().numpy()[0,:],'--')
    plt.subplot(313)
    plt.plot(weight[...,args.nrecept-1:].detach().cpu().numpy()[0,:])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(filename)


def train_seq(data, target, weight, model, optimizer, args):
    '''Takes a batch sequence and trains, splitting if needed'''
    if args.cuda: 
        data, target, weight  = data.cuda(non_blocking=True), \
                                target.cuda(non_blocking=True), \
                                weight.cuda(non_blocking=True)
    data = data.view(data.shape[0], args.input_channels, -1)
    
    #No data splitting
    optimizer.zero_grad()
    output = model(data)
    #do mean of loss by hand to handle unequal sequence lengths
    loss = F.binary_cross_entropy(output[...,args.nrecept-1:],target[...,args.nrecept-1:],weight=weight[...,args.nrecept-1:])
    loss.backward()
    if args.clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    #split data into subsequences to process
    #train_loss = process_seq(data,target,args.nsub,args.nrecept,model,
    #                          optimizer=optimizer,weight=weight,
    #                          train=True,clip=args.clip,accumulate=args.accumulate)
    return loss.item()


def process_seq(data,target,Nsub,Nrecept,model,optimizer=None,train=True,weight=None,clip=None,accumulate=False):
    '''Splits apart sequence into equal, overlapping subsequences of length Nsub, with overlap Nrecept
    If accumulate=True, does accumulated gradients method to avoid large GPU memory usage
    '''
    if weight is None: weight = torch.ones(target.shape).cuda()
    N = data.shape[-1] #length of entire sequence
    num_seq_frac = (N - Nsub)/float(Nsub - Nrecept + 1)+1 #this assumes N>=Nrecept
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

        if (optimizer is not None) & ((m==0) or (not accumulate)):
            optimizer.zero_grad()
        ys = model(data[...,start_idx:stop_idx])
        ts = target[...,start_idx:stop_idx]
        ws = weight[...,start_idx:stop_idx]
        #do mean of loss by hand to handle unequal sequence lengths
        loss = F.binary_cross_entropy(ys[...,Nrecept-1:],ts[...,Nrecept-1:],weight=ws[...,Nrecept-1:],reduction='sum')/(N-Nrecept+1)
        ####REMOVE ME
        #if accumulate:
        #    loss = loss/num_seq
        ####END REMOVE ME
        if train: loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        total_losses += loss.item()
        if (optimizer is not None) & (not accumulate):
            optimizer.step()
    if accumulate:
        optimizer.step()
    return total_losses


#for the validation and test set
def evaluate(val_loader,model,args):
    model.eval()
    total_loss = 0
    total = 0
    correct = np.zeros(args.thresholds.shape)
    TPs = np.zeros(args.thresholds.shape)
    TP_FPs = np.zeros(args.thresholds.shape)
    TP_FNs = np.zeros(args.thresholds.shape)
    with torch.no_grad(): #turns off backprop, saves computation
        for batch_idx,(data, target,global_index,weight) in enumerate(val_loader):
            data, target, weight = data.cuda(non_blocking=True), target.cuda(non_blocking=True), weight.cuda(non_blocking=True)
            data = data.view(data.shape[0], args.input_channels, -1)
            output = model(data)
            loss = F.binary_cross_entropy(output[...,args.nrecept-1:], target[...,args.nrecept-1:], weight=weight[...,args.nrecept-1:]).item()
            total_loss += loss
            total += target[...,args.nrecept-1:].numel()
            for i,threshold in enumerate(args.thresholds):
                correct[i] += accuracy(output[...,args.nrecept-1:],target[...,args.nrecept-1:],threshold=threshold) 
                TP, TP_FP, TP_FN = f1_score_pieces(output[...,args.nrecept-1:],target[...,args.nrecept-1:],threshold=threshold)
                TPs[i] += TP; TP_FPs[i] += TP_FP; TP_FNs[i] += TP_FN
            
            #plot disruptive output
            for (i,gi) in enumerate(global_index):
                if ((val_loader.dataset.dataset.disruptedi[gi]==1)):
                    plot_output(data,output,target,weight,args,
                            filename='output_'+str(int(os.environ['SLURM_JOB_ID']))+'_iteration_'+str(args.iteration)+'_ind_'+str(int(gi))+'.png',
                            title='Loss: %0.4e' % float(loss))

        total_loss /= len(val_loader)
        if args.distributed:
            dist.all_reduce(torch.tensor(total_loss), op=dist.ReduceOp.SUM)
            dist.all_reduce(torch.tensor(correct), op=dist.ReduceOp.SUM)
            dist.all_reduce(torch.tensor(total), op=dist.ReduceOp.SUM)
            dist.all_reduce(torch.tensor(TPs), op=dist.ReduceOp.SUM)
            dist.all_reduce(torch.tensor(TP_FPs), op=dist.ReduceOp.SUM)
            dist.all_reduce(torch.tensor(TP_FNs), op=dist.ReduceOp.SUM)
            total_loss = total_loss.item()/args.world_size; total = total.item()
            correct = correct.numpy(); TPs = TPs.numpy()
            TP_FPs = TP_FPs.numpy(); TP_FNs = TP_FNs.numpy()
        if args.rank==0:
            f1 = f1_score(TPs,TP_FPs,TP_FNs)
            f1max = np.nanmax(f1)
            correctmax = np.nanmax(correct).astype(int)
            print('\nValidation set: Average loss: {:.6e}, Accuracy: {}/{} ({:.0f}%), F1: {:.6e}\n'.format(
                    total_loss, correctmax, total,
                    100. * correctmax / total, f1max))
        return total_loss,correctmax/total, f1max

def accuracy(output,target,threshold=0.5):
    pred = output.ge(threshold)
    return pred.eq(target.type_as(pred).view_as(pred)).cpu().float().sum()

def f1_score_pieces(output,target,threshold=0.5):
    pred = output.ge(threshold).type_as(target)
    TP = (pred*target).cpu().float().sum()
    TP_FP = pred.cpu().sum()
    TP_FN = target.cpu().sum()
    return TP, TP_FP, TP_FN 

def f1_score(TP,TP_FP,TP_FN,eps=1e-10):
    precision = TP/TP_FP+eps
    recall = TP/TP_FN+eps
    return 2./(1./precision + 1./recall)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.'+str(int(os.environ['SLURM_JOB_ID']))+'.pth.tar')


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


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        
    def update(self, val):
        dist.all_reduce(val, op=dist.reduce_op.SUM)
        self.sum += val
        self.n += 1
        
    @property
    def avg(self):
        return self.sum / self.n

if __name__ == "__main__":
    tstart = time.time()
    main()
    #TODO: test data set, create final statistics (ROC? Printed recall/precision?)
