import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from disruptcnn.loader import data_generator, EceiDataset
from disruptcnn.model import TCN

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
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

args.distributed = args.world_size > 1 or args.multiprocessing_distributed

if not args.distributed: args.workers = 0

root = '/scratch/gpfs/rmc2/ecei_d3d/'
data_root = root+'data/'
clear_file = root + 'd3d_clear_ecei.final.txt'
disrupt_file = root + 'd3d_disrupt_ecei.final.txt'
batch_size = 1#args.batch_size
n_classes = 1 #for binary classification
input_channels = 160
epochs = args.epochs
steps = 0
#seq_length = int(Nseq) #TODO: This might be different from how I defined it in loader.py
#TODO remove, use general passed in instead
args.nhid = 20
args.levels = 5
dilation_sizes = [1,10,100,1000,6783] #dilation=10, except last which is set to give receptive field ~Nmodel=300,000
Nsub = 1000000 #found by taking receptive field, and scaling for 15GB of GPU memory #TODO automate
Nrecept = 300000
args.log_interval = 1
#TODO have to write overlapping sequences, not just put in

print(args)
dataset = EceiDataset(data_root,clear_file,disrupt_file)
#create the indices for train/val/test split
dataset.train_val_test_split()
#create data loaders
train_loader, val_loader, test_loader = data_generator(dataset, batch_size, 
                                                        distributed=args.distributed,
                                                        num_workers=args.workers)

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, 
            kernel_size=kernel_size, 
            dropout=args.dropout,
            dilation_size=dilation_sizes)

if args.cuda:
    model.cuda()

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

lr = args.lr
#optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

def train(epoch):
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                        epoch, batch_idx * batch_size, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss.data[0]/args.log_interval, steps))
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


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        evaluate(epoch)
        #TODO: Determine if I want this LR decrease, or automated ways (momentum?)
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    #TODO: test data set, create final statistics (ROC? Printed recall/precision?)
