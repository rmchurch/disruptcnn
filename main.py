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
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = '/gpfs/rchurchi/ecei_d3d/'
clear_file = root + 'd3d_clear_ecei.final.txt'
disrupt_file = root + 'd3d_disrupt_ecei.final.txt'
batch_size = args.batch_size
n_classes = 2
input_channels = 160
seq_length = int(Nseq) #TODO: This might be different from how I defined it in loader.py
epochs = args.epochs
steps = 0

print(args)
dataset = EceiDataset(root,clear_file,disrupt_file)
#create the indices for train/val/test split
dataset.train_val_test_split()
#create data loaders
train_loader, val_loader, test_loader = data_generator(dataset, batch_size, 
                                                        distributed=args.distributed,
                                                        num_workers=args.workers)

channel_sizes = [args.nhid] * args.levels
#TODO: dilation optional input
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()

lr = args.lr
#optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

def train(epoch):
    global steps
    train_loss = 0
    model.train()
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
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
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss += F.nll_loss(output, target, size_average=False).data[0]
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