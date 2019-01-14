import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from disruptcnn.loader import data_generator, EceiDataset
from disruptcnn.model import TCN
import time

parser = argparse.ArgumentParser(description='Sequence Modeling - disruption ECEi')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size (default: 8)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: None)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')



#TODO Generalize
root = '/scratch/gpfs/rmc2/ecei_d3d/'
data_root = root+'data/'
clear_file = root + 'd3d_clear_ecei.final.txt'
disrupt_file = root + 'd3d_disrupt_ecei.final.txt'
batch_size = 1#args.batch_size
n_classes = 1 #for binary classification
input_channels = 160
steps = 0
#seq_length = int(Nseq) #TODO: This might be different from how I defined it in loader.py
dilation_sizes = [1,10,100,1000,6783] #dilation=10, except last which is set to give receptive field ~Nmodel=300,000
Nsub = 1000000 #found by taking receptive field, and scaling for 15GB of GPU memory #TODO automate
Nrecept = 300000



def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

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
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    #TODO remove, use general passed in instead
    epochs = args.epochs
    args.nhid = 20
    args.levels = 5
    args.log_interval = 1
    #TODO have to write overlapping sequences, not just put in

    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    model = TCN(input_channels, n_classes, channel_sizes, 
                kernel_size=kernel_size, 
                dropout=args.dropout,
                dilation_size=dilation_sizes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
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
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print(args)
    dataset = EceiDataset(data_root,clear_file,disrupt_file)
    #create the indices for train/val/test split
    dataset.train_val_test_split()
    #create data loaders
    train_loader, val_loader, test_loader = data_generator(dataset, batch_size, 
                                                            distributed=args.distributed,
                                                            num_workers=args.workers)

    lr = args.lr
    #optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        #TODO: Determine if I want this LR decrease, or automated ways (momentum?)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = evaluate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

def train(train_loader, model, optimizer, epoch, args):
    global steps #TODO: why do I need steps??
    train_loss = 0
    model.train()
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.cuda: data, target = data.cuda(), target.cuda() #TODO: verify can put ENTIRE sequence on GPU
        data = data.view(batch_size, input_channels, -1)
        #split data into subsequences to process
        train_loss += process_seq(data,target,Nsub)
        steps += data.shape[-1]
        optimizer.step()
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}\tTime: {}'.format(
                        epoch, batch_idx * batch_size, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss/args.log_interval, steps,(time.time()-tstart)))
            train_loss = 0


#for the validation and test set
def evaluate(epoch):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad(): #turns off backprop, saves computation
        for data, target in val_loader:
            if args.cuda: data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            output = model(data)
            loss += F.binary_cross_entropy(output, target, size_average=False).data[0]
            #TODO: Enter validation loss here, modify from this simple
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return loss


def process_seq(x,target,Nsub):
    '''Splits apart sequence into equal, overlapping subsequences of length Nsub, with overlap Nrecept
    Does accumulated gradients method to avoid large GPU memory usage
    '''
    N = x.shape[-1] #length of entire sequence
    num_seq_frac = (N - Nsub)/float(Nsub - Nrecept + 1)+1
    num_seq = np.ceil(num_seq_frac).astype(int)
    total_losses = 0
    for m in range(num_seq):
        start_idx =    m*Nsub - m*Nrecept + m
        stop_idx = (m+1)*Nsub - m*Nrecept + m
        if stop_idx>N: stop_idx = N
        if ((stop_idx-start_idx)<Nrecept):
            start_idx = stop_idx - Nrecept
        ys = model(x[...,start_idx:stop_idx])
        ts = target[...,start_idx:stop_idx]
        #do mean of loss by hand to handle unequal sequence lengths
        loss = F.binary_cross_entropy(ys[...,Nrecept-1:],ts[...,Nrecept-1:],reduction='sum')/(N-Nrecept+1)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        total_losses += loss.item()
    return total_losses


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    tstart = time.time()
    main()
    #TODO: test data set, create final statistics (ROC? Printed recall/precision?)
