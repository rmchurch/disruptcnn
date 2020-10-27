#!/usr/bin python
import random
import warnings
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
import numpy as np
import argparse
from disruptcnn.loader import data_generator, EceiDataset, data_prefetcher
from disruptcnn.model import TCN
import time
from tensorboardX import SummaryWriter
import os, psutil, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle
#apex
import apex
from apex import amp
#pyprof
import torch.cuda.profiler as profiler
#import pyprof
#pyprof.init()


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
parser.add_argument('--lr-scheduler', type=str, default='plateau',
                    help='Type of learning rate scheduler (default: "plateau", other valid option "step")')
parser.add_argument('--lr-step-metric', type=str, default='valid_f1',
                    help='Metric to use with ReduceLROnPlateau (default: valid_f1)')
parser.add_argument('--lr-factor', type=float, default=0.5,
                    help='learning rate reduction factor when change with ReduceLROnPlateau (default: 0.5)')
parser.add_argument('--lr-patience', type=int, default=20,
                    help='learning rate wait period before change with ReduceLROnPlateau (default: 20)')
parser.add_argument('--lr-cooldown', type=int, default=10,
                    help='learning rate wait period after change with ReduceLROnPlateau (default: 10)')
parser.add_argument('--lr-epochs', default=[100,150,200,250], nargs='*',type=int,
                    help='list of epochs which to decay by lr-factor, used with "step" option of lr-scheduler (MultiStepLR) (default: [100,150,200,250])')

parser.add_argument('--weight-decay', type=float, const=1e-4, nargs='?',default=0.0,
                    help='weight-decay, acts as L2 regularizer (default: None if no floag, 1e-4 if flag but no value)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--epochs-valid', type=int, default=10,
                    help='epoch period to run validation (default: 10)')
parser.add_argument('--iterations-valid', type=int, const=200, nargs='?',
                    help='iteration period to run validation, if set overrides epochs-valid (200 iterations if flag but no value)')
parser.add_argument('--epochs-warmup', type=int, default=5,
                    help='LR warmup epochs (default: 5 epochs)')
parser.add_argument('--iterations-warmup', type=int, const=200, nargs='?',
                    help='LR warmup iterations, if set overrides epochs-warmup (default: 200 iterations if flag but no value)')
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
parser.add_argument('--oversample', type=int, nargs='?',const=5,
                    help='''Duplicate pos samples, match with equal neg samples (default: None if no flag, 5 if flag but no value) 
                               (e.g. if total N = Npos + Nneg, and Nneg/Npos = 10, oversample=2 will set the # positive samples as 2*Npos, and
                               # negative samples as 2*Npos (i.e. 5x smaller than available) )''')
#other
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')
parser.add_argument('--log-interval', type=int, const=100, nargs='?',
                    help='Iteration frequency of logging (default: every epoch or test if no flag, 100 iterations if flag but no value)')
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
parser.add_argument('--plot', action='store_true',
                    help='plot validation disruptive sequences (default: False)')
parser.add_argument('--flattop-only', action='store_true',
                    help='use only data from the current flattop (default: False)')
parser.add_argument('--amp', action='store_true',
                    help='use automatic mixed-precision (default: False)')
parser.add_argument('--opt-level', default='O1', type=str,
                    help='Optimization level for AMP (default: O1)')



root = '/gpfs/alpine/proj-shared/fus131/ecei_d3d/'
data_root = root+'data/'
clear_file = root + 'd3d_clear_ecei.final.txt'
disrupt_file = root + 'd3d_disrupt_ecei.final.txt'



def main():
    args = parser.parse_args()

    #assert (args.batch_size==1), "Currently need batch_size=1, due to variable length sequences"
    assert torch.cuda.is_available(), "GPU is currently required"

    args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
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
	#not sure if this is right, Pytorch may expect actual GPU node id, which I may need to get with JSM_GPU_ASSIGNMENTS.
	#using LOCAL_RANK here is assuming 1 GPU per rank (jsrun --gpu_per_rs 1) 
	#->actually, with --gpu_per_rs 1, there will only be 1 torch.cuda.device_count(), so all should be 0.
        args.gpu = 0 #int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) 
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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,weight_decay=args.weight_decay)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    if args.amp: amp.register_float_function(torch, 'sigmoid') #needed because F.binary_cross_entropy
    model, optimizer = amp.initialize(model, optimizer, enabled=args.amp, opt_level=args.opt_level)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            #TODO: I am making batch_size per process. Generalize?
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            #args.batch_size = int(args.batch_size / ngpus_per_node)
            #args.workers = int(args.workers / ngpus_per_node)
            if args.amp:
                if args.rank==0: print("WARNING: amp for DistributedDataParallel assumes 1:1 GPU to process")
                model = apex.parallel.DistributedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            if args.amp:
                model = apex.parallel.DistributedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model.cuda()
        if args.amp:
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
        # DataParallel will divide and allocate batch_size to all available GPUs
        # model = torch.nn.DataParallel(model).cuda()

    if (args.test>0) and (args.test < args.batch_size): args.batch_size = args.test

    dataset = EceiDataset(data_root,clear_file,disrupt_file,
                          test=args.test,test_indices=args.test_indices,
                          label_balance=args.label_balance,
                          normalize=(not args.no_normalize),
                          data_step=args.data_step,
                          nsub=args.nsub,nrecept=args.nrecept,
                          flattop_only=args.flattop_only)
    #create the indices for train/val/test split
    dataset.train_val_test_split()
    #create data loaders
    train_loader, val_loader, test_loader = data_generator(dataset, args.batch_size, 
                                                            distributed=args.distributed,
                                                            num_workers=args.workers,
                                                            undersample=args.undersample,
                                                            oversample=args.oversample)

    #set defaults for iterations_warmup (5 epochs) and iterations_valid (1 epoch)
    #TODO Add separate argsparse for epochs_warmup and epochs_valid?
    if args.iterations_warmup is None: args.iterations_warmup = args.epochs_warmup*len(train_loader)
    if args.iterations_valid is None: args.iterations_valid = args.epochs_valid*len(train_loader)
    if args.log_interval is None: 
        args.log_interval = len(train_loader)

    #TODO Generalize
    args.thresholds = np.linspace(0.05,0.95,19)

    #TODO generalize momentum?
    #TODO implement general optimizer
    if not args.lr_finder:
        #gradual linear increasing learning rate for warmup
        lambda1=lambda iteration: (1.-1./args.multiplier_warmup)/args.iterations_warmup*(iteration+1) + 1./args.multiplier_warmup
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)
        #decaying learning rate scheduler for after warmup
        #TODO generalize factor?
        if 'f1' in args.lr_step_metric:
            mode = 'max'
        else:
            mode = 'min'
        if 'plateau' in args.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=args.lr_factor,
                                                                       min_lr=args.lr*args.lr_factor**4,
                                                                       patience=args.lr_patience,
                                                                       cooldown=args.lr_cooldown,
                                                                       mode=mode,
                                                                       threshold=0.01)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                [lr_epoch*len(train_loader) for lr_epoch in args.lr_epochs],
                                gamma=args.lr_factor)
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
            slurm_resume_id = ''.join(filter(str.isdigit, args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            #if args.gpu is not None:
            #    # best_acc may be from a checkpoint from a different GPU
            #    best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            #load same splits
            fsplits = np.load('splits.'+slurm_resume_id+'.npz')
            train_inds = fsplits['train_inds']
            val_inds = fsplits['val_inds']
            test_inds = fsplits['test_inds']

            #recreate the loaders with the splits from before
            dataset.train_val_test_split(train_inds=train_inds,val_inds=val_inds,test_inds=test_inds)
            
            #NOTE: to reuse the train_inds, etc. as defined by the splits file, the undersample has to
            #      be turned off here
            #      But oversample has to be on if on, otherwise uses dataset as is
            train_loader, val_loader, test_loader = data_generator(dataset, args.batch_size, 
                                                        distributed=args.distributed,
                                                        num_workers=args.workers,
                                                        undersample=None, oversample=args.oversample)
            #fast-forward scheduler to be at right epoch
            if 'step' in args.lr_scheduler:
                for i in range(args.start_epoch*len(train_loader)): scheduler.step()

            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    #save the train/val/test split, for further post-processing
    if args.rank==0:
        print(args)
        if getattr(val_loader,'pos_used_indices',None) is not None:
            val_pos_used_indices=val_loader.sampler.pos_used_indices
            val_neg_used_indices=val_loader.sampler.neg_used_indices
        else:
            val_pos_used_indices=dataset.val_inds[dataset.disruptedi[dataset.val_inds]==1]
            val_neg_used_indices=dataset.val_inds[dataset.disruptedi[dataset.val_inds]==0]

        np.savez('splits.'+os.environ['LSB_JOBID']+'.npz',
                    shot=dataset.shot,shot_idxi=dataset.shot_idxi,start_idxi=dataset.start_idxi,stop_idxi=dataset.stop_idxi,
                    disrupted=dataset.disrupted,disruptedi=dataset.disruptedi,
                    train_inds = dataset.train_inds,val_inds = dataset.val_inds, test_inds=dataset.test_inds,
                    train_pos_used_indices=train_loader.sampler.pos_used_indices,
                    train_neg_used_indices=train_loader.sampler.neg_used_indices,
                    val_pos_used_indices=val_pos_used_indices,
                    val_neg_used_indices=val_neg_used_indices,
                    test_pos_used_indices=dataset.test_inds[dataset.disruptedi[dataset.test_inds]==1],
                    test_neg_used_indices=dataset.test_inds[dataset.disruptedi[dataset.test_inds]==0])


    #this autotunes algo on GPU. If variable input (like before with single shot), would
    #be worse performance
    cudnn.benchmark = True


    #main training loop
    steps = 0
    total_loss_lr = 0
    total_loss_log = 0
    best_f1 = 0
    valid_f1 = 0
    val_iterator = cycle(val_loader) #cycle to cache data, since small for validation
    for epoch in range(args.start_epoch, args.epochs):
        nvtx.range_push("Epoch "+str(epoch))
        train_loader.sampler.set_epoch(epoch)

        nvtx.range_push("Init prefetch "+str(epoch))
        if True:#epoch==0:
            prefetcher = data_prefetcher(train_loader)
            #prefetcher = iter(train_loader)
            #prefetcher = cycle(train_loader)
        #else:
        #    prefetcher = next_prefetcher
        nvtx.range_pop()
        for batch_idx in range(len(train_loader)):
            #if batch_idx==int(len(train_loader)/2):
            #    nvtx.range_push("Init next_prefetch "+str(epoch))
            #    next_prefetcher = data_prefetcher(train_loader)
            #    nvtx.range_pop()
            nvtx.range_push("Data Load "+str(batch_idx))
            #data, target, global_index, weight = next(prefetcher)#.next()
            data, target, global_index, weight = prefetcher.next()
            nvtx.range_pop()

            nvtx.range_push("Batch "+str(batch_idx))
            model.train()
            iteration = epoch*len(train_loader) + batch_idx
            args.iteration = iteration


            #train single iteration
            train_loss = train_seq(data,target,weight,model,optimizer,args)
            if is_writer: writer.add_scalar('train_loss',train_loss,iteration)
            steps += data.shape[0]*data.shape[-1]
            with torch.no_grad():
                total_loss_lr += train_loss
                total_loss_log += train_loss

            #validation
            if (iteration>0) & (iteration % args.iterations_valid == 0) & (args.test==0):
                valid_loss, valid_acc, valid_f1, TP, TN, FP, FN,threshold = evaluate(val_iterator, model, args, len(val_loader))
                
                if is_writer: 
                    writer.add_scalar('valid_loss',valid_loss,iteration)
                    writer.add_scalar('valid_acc',valid_acc,iteration)
                    writer.add_scalar('valid_f1',valid_f1,iteration)
                # remember best f1 and save checkpoint
                is_best = valid_f1 > best_f1
                best_f1 = max(valid_f1, best_f1)
                 
                if (not args.multiprocessing_distributed and args.rank==0) or \
                   (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    check_file = 'checkpoint.'+os.environ['LSB_JOBID']+'.epoch.'+str(epoch+1)+'.pth.tar'
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_f1': best_f1,
                        'optimizer' : optimizer.state_dict(),
                        'args': args,
                        'confusion_matrix': {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN},
                        'f1': valid_f1,
                        'threshold': threshold,
                    }, is_best,filename=check_file)
            
            #log training 
            if (iteration>0) & (iteration % args.log_interval == (args.log_interval-1)):
                #TODO: may want to accumulate Ndisrupted and Ntotal if log-interval not 1
                Ndisrupted = np.sum(train_loader.dataset.dataset.disruptedi[global_index])
                Ntotal = global_index.size()
                if args.distributed: 
                    total_loss_log = all_reduce(total_loss_log).item()/args.world_size/args.log_interval
                    #TEST-find the true number of disruptive seqs in the distributed batch
                    Ndisrupted = all_reduce(Ndisrupted).item()
                    Ntotal = all_reduce(Ntotal).item()
                fracDisrupted = Ndisrupted/Ntotal 

                if args.rank==0:
                    lr_epoch = [ group['lr'] for group in optimizer.param_groups ][0]
                    print('Train Epoch: %d [%d/%d (%0.2f%%)]\tIteration: %d\tDisrupted: %0.4f\tLoss: %0.6e\tSteps: %d\tTime: %0.2f\tMem: %0.1f\tLR: %0.2e' % (
                                epoch, batch_idx+1, len(train_loader), 100. * ((batch_idx+1) / len(train_loader)), iteration,
                                fracDisrupted, total_loss_log, steps,(time.time()-args.tstart),psutil.virtual_memory().used/1024**3.,lr_epoch))
                total_loss_log = 0

            #learning rate scheduler
            if args.lr_finder:
                if (iteration>0) and (iteration % niter_per_interval == 0):
                    scheduler_lrfinder.step()
                    lr_epoch = [ group['lr'] for group in optimizer.param_groups ][0]
                    lr_history["lr"].append(lr_epoch)
                    lr_history["loss"].append(total_loss_lr)
                    np.savez('lr_finder_'+str(int(os.environ['LSB_JOBID']))+'.npz',lr=lr_history["lr"],loss=lr_history["loss"])
                    total_loss_lr = 0*total_loss_lr
            else:
                if iteration < args.iterations_warmup:
                    scheduler_warmup.step(iteration)
                    if 'train' in args.lr_step_metric:
                        total_loss_lr = 0
                else:
                    #TODO change to be general outside of test
                    if args.test==0:
                        if 'plateau' in args.lr_scheduler:
                            if 'valid' in args.lr_step_metric:
                                if (iteration>0) and (iteration % args.iterations_valid == (args.iterations_valid-1)):
                                    if 'valid_f1' in args.lr_step_metric:
                                        metric = valid_f1
                                    else: #'valid_loss'
                                        metric = valid_loss
                                    scheduler.step(metric)
                            else:
                                if (iteration>0) and (iteration % len(train_loader) == 0):
                                    #validation metrics are already reduced across world_size, train_loss not
                                    if args.distributed: 
                                        total_loss_lr = all_reduce(total_loss_lr).item()
                                        total_loss_lr = total_loss_lr/args.world_size/args.iterations_valid
                                    metric = total_loss_lr
                                    scheduler.step(metric)
                                    total_loss_lr = 0
                        else:
                            scheduler.step()
                    else:
                        if (iteration>0) and (iteration % len(train_loader) == 0):
                            scheduler.step(total_loss_lr)
    
            
            nvtx.range_pop() #end batch nvtx
        nvtx.range_pop() #end epoch nvtx
            

    print("Main training loop ended: Rank: %d, Time: %0.2f" % (args.rank,(time.time()-args.tstart)))


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
                           filename='test_output_'+str(int(os.environ['LSB_JOBID']))+'_ind_'+str(global_index.item())+'.png',
                           title='Loss: %0.4e' % loss)

    if args.lr_finder:
        plt.figure()
        plt.plot(lr_history["lr"],lr_history["loss"])
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim([np.array(lr_history["loss"]).min(),lr_history["loss"][0]])
        plt.savefig('lr_finder_'+str(int(os.environ['LSB_JOBID']))+'.png')
        np.savez('lr_finder_'+str(int(os.environ['LSB_JOBID']))+'.npz',lr=lr_history["lr"],loss=lr_history["loss"])

    if is_writer: writer.close()
    time.sleep(180) #allow all processes to finish

def all_reduce(data,op=dist.ReduceOp.SUM):
    data = torch.as_tensor(data).cuda()
    dist.all_reduce(data,op=op)
    return data

def plot_output(data,output,target,weight,args,filename='output.png',title=''):
    plt.clf()
    plt.subplot(311)
    plt.plot(data[0,:].detach().cpu().numpy())
    plt.title(title)
    plt.subplot(312)
    plt.plot(output[...,args.nrecept-1:].detach().cpu().numpy())
    plt.plot(target[...,args.nrecept-1:].detach().cpu().numpy(),'--')
    plt.subplot(313)
    plt.plot(weight[...,args.nrecept-1:].detach().cpu().numpy())
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(filename)


def train_seq(data, target, weight, model, optimizer, args):
    '''Takes a batch sequence and trains, splitting if needed'''
    nvtx.range_push("Copy to device (train_seq)")
    if args.cuda: 
        data, target, weight  = data.cuda(non_blocking=True), \
                                target.cuda(non_blocking=True), \
                                weight.cuda(non_blocking=True)
    nvtx.range_pop()
    nvtx.range_push("Data reorder (train_seq)")
    data = data.view(data.shape[0], args.input_channels, -1)
    nvtx.range_pop()
    
    #No data splitting
    nvtx.range_push("Forward pass (train_seq)")
    optimizer.zero_grad()
    output = model(data)
    #do mean of loss by hand to handle unequal sequence lengths
    loss = F.binary_cross_entropy(output[...,args.nrecept-1:],target[...,args.nrecept-1:],weight=weight[...,args.nrecept-1:])
    nvtx.range_pop()
    nvtx.range_push("Backward pass + optimizer step (train_seq)")
    with amp.scale_loss(loss,optimizer) as scaled_loss:
        scaled_loss.backward()
    if args.clip is not None:
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
    optimizer.step()
    nvtx.range_pop()

    #split data into subsequences to process
    #train_loss = process_seq(data,target,args.nsub,args.nrecept,model,
    #                          optimizer=optimizer,weight=weight,
    #                          train=True,clip=args.clip,accumulate=args.accumulate)
    return loss


#for the validation and test set
def evaluate(val_iterator,model,args,len_val_loader):
    model.eval()
    total_loss = 0
    total = torch.tensor(0).cuda()
    correct = torch.zeros(args.thresholds.shape)
    TPs = torch.zeros(args.thresholds.shape)
    TNs = torch.zeros(args.thresholds.shape)
    FPs = torch.zeros(args.thresholds.shape)
    FNs = torch.zeros(args.thresholds.shape)
    if 'nccl' in args.backend:
        correct = correct.cuda()
        TPs = TPs.cuda()
        TNs = TNs.cuda()
        FPs = FPs.cuda()
        FNs = FNs.cuda()
    with torch.no_grad(): #turns off backprop, saves computation
        for batch_idx in range(len_val_loader):
            data, target,global_index,weight = next(val_iterator)
            data, target, weight = data.cuda(non_blocking=True), target.cuda(non_blocking=True), weight.cuda(non_blocking=True)
            data = data.view(data.shape[0], args.input_channels, -1)
            output = model(data)
            loss = F.binary_cross_entropy(output[...,args.nrecept-1:], target[...,args.nrecept-1:], weight=weight[...,args.nrecept-1:])
            total_loss += loss
            total += target[...,args.nrecept-1:].numel()
            for i,threshold in enumerate(args.thresholds):
                correct[i] += accuracy(output[...,args.nrecept-1:],target[...,args.nrecept-1:],threshold=threshold) 
                TP, TN, FP, FN = confusion_matrix(output[...,args.nrecept-1:],target[...,args.nrecept-1:],threshold=threshold)
                TPs[i] += TP; TNs[i] += TN; FPs[i] += FP; FNs[i] += FN
            
            #plot disruptive output
            if args.plot:
                for (i,gi) in enumerate(global_index):
                    if ((target[i,...].max()>0)):
                        plot_output(data[i,...],output[i,...],target[i,...],weight[i,...],args,
                                filename='output_'+str(int(os.environ['LSB_JOBID']))+'_iteration_'+str(args.iteration)+'_ind_'+str(int(gi))+'.png',
                                title='Loss: %0.4e' % float(loss))

        total_loss /= len_val_loader
        if args.distributed:
            #print('Before all_reduce, Rank: ',str(args.rank),' Correct: ',*correct, ' Correct type: ',type(correct), 'Time: ',(time.time()-args.tstart))
            total_loss = all_reduce(total_loss)
            correct = all_reduce(correct)
            total = all_reduce(total)
            TPs = all_reduce(TPs)
            TNs = all_reduce(TNs)
            FPs = all_reduce(FPs)
            FNs = all_reduce(FNs)
            total_loss = total_loss/args.world_size
        total_loss = total_loss.item()
        correct = correct.cpu().numpy()
        total = total.item()
        TPs = TPs.cpu().numpy()
        TNs = TNs.cpu().numpy()
        FPs = FPs.cpu().numpy()
        FNs = FNs.cpu().numpy()
            #print('After all_reduce, Rank: ',str(args.rank),' Correct: ',*correct, ' Correct type: ',type(correct), 'Time: ',((time.time()-args.tstart)))
        f1 = f1_score(TPs,TPs+FPs,TPs+FNs)
        f1max = np.nanmax(f1)
        mcc = mcc_score(TPs,FPs,TNs,FNs)
        mccmax = np.nanmax(mcc)
        thresholdmax = args.thresholds[np.nanargmax(f1)]
        tpr = (TPs/(TPs+FNs))[np.nanargmax(f1)]
        fpr = (TNs/(TNs+FPs))[np.nanargmax(f1)]
        #
        correctmax = np.nanmax(correct).astype(int)
        if args.rank==0:
            print('\nValidation set [{}]:\tAverage loss: {:.6e}\tAccuracy: {:.6e} ({}/{})\tTPR: {:.6e}\tFPR: {:.6e}\tF1: {:.6e}\tMCC: {:.6e}\tThreshold: {:.2f}\tTime: {:.2f}\n'.format(
                    len_val_loader,total_loss,
                    correctmax / total, correctmax, total, tpr, fpr, f1max,mccmax,thresholdmax,(time.time()-args.tstart)))
        return total_loss,correctmax/total, f1max, TPs, TNs, FPs, FNs, thresholdmax


def confusion_matrix(output,target,threshold=0.5):
    pred = output.ge(threshold).type_as(target)
    TP = ((pred==1) & (target==1)).float().sum()
    TN = ((pred==0) & (target==0)).float().sum()
    FP = ((pred==1) & (target==0)).float().sum()
    FN = ((pred==0) & (target==1)).float().sum()
    return TP,TN,FP,FN

def accuracy(output,target,threshold=0.5):
    pred = output.ge(threshold)
    return pred.eq(target.type_as(pred).view_as(pred)).float().sum()

def f1_score_pieces(output,target,threshold=0.5):
    pred = output.ge(threshold).type_as(target)
    TP = (pred*target).float().sum()
    TP_FP = pred.sum()
    TP_FN = target.sum()
    return TP, TP_FP, TP_FN 

def f1_score(TP,TP_FP,TP_FN,eps=1e-10):
    precision = TP/TP_FP+eps
    recall = TP/TP_FN+eps
    return 2./(1./precision + 1./recall)

def mcc_score(TP,FP,TN,FN):
    denom = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if np.any(denom==0): 
        denom[denom==0] = 1 
    mcc = ((TP*TN) - (FP*FN))/denom
    return mcc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.'+str(int(os.environ['LSB_JOBID']))+'.pth.tar')


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
                dilation_size=dilation_sizes,nsub=args.nsub)
    return model

def calc_seq_length(kernel_size,dilation_sizes,nlevel):
    """Assumes kernel_size scalar, dilation_size exponential increase"""
    if np.isscalar(dilation_sizes): dilation_sizes = dilation_sizes**np.arange(nlevel)
    return 1 + 2*(kernel_size-1)*np.sum(dilation_sizes)


if __name__ == "__main__":
    tstart = time.time()
    main()
