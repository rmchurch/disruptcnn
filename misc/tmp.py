#!/usr/bin python

#this is just to test out different sequence lengths, to figure out what will fit into GPU memory
from model import TCN
import numpy as np
from torchsummary import summary

from scipy.optimize import fsolve
def funcLevel(x,*args):
     S,k,d = args
     return 1 + 2*(k-1)*(d**x-1.)/(d-1.) - S

root = './data/mnist'
batch_size = 1
n_classes = 2
input_channels = 160
kernel_size = 20
nhid = 25
dropout = 0.05

def calc_seq_length(kernel_size,dilation_sizes,nlevel):
     """Assumes kernel_size and dilation_size scalar, dilation_size exponential increase"""
     if np.isscalar(dilation_sizes): dilation_sizes = dilation_sizes**np.arange(nlevel)
     return 1 + 2*(kernel_size-1)*np.sum(dilation_sizes)

sequence_length = 300000
total_sizes = []
seq_lengths = []
for d in range(4,16,2)[::-1]:
     levelfrac = fsolve(funcLevel,30,args=(sequence_length,kernel_size,d))
     level = int(np.ceil(levelfrac))
     channel_sizes = [nhid] * level
     seq_length = calc_seq_length(kernel_size,d,level)
     last_dilation = int(np.ceil((sequence_length - calc_seq_length(kernel_size,d,level-1))/(2*(kernel_size-1))))
     dilation_sizes = (d**np.arange(level-1)).tolist() + [last_dilation]
     #dilation_sizes = (d**np.arange(level)).tolist()
     seq_length = calc_seq_length(kernel_size,dilation_sizes,level)
     #print d,seq_length,level
     #continue
     model = TCN(input_channels, n_classes, channel_sizes, dilation_size=dilation_sizes,kernel_size=kernel_size, dropout=dropout)
     s,sp = summary(model,input_size=(input_channels,2*seq_length),batch_size=batch_size,noprint=True)
     total_sizes += [sp['total_size']]
     seq_lengths += [seq_length]
     print(d,seq_length,level,sp['total_size'], last_dilation)

# level = 8
# dilation_size = 2
# seq_length = calc_seq_length(kernel_size,dilation_size,level)
# channel_sizes = [nhid] * level
# print "****Setup 1****"
# print "nlevel: %d" % level
# print "dilation_size: %d" % dilation_size
# print "kernel_size: %d" % kernel_size
# print "Sequence length: %d" % seq_length
# model = TCN(input_channels, n_classes, channel_sizes, dilation_size=dilation_size,kernel_size=kernel_size, dropout=dropout)
# s,sp = summary(model,input_size=(input_channels,seq_length),batch_size=batch_size,print_reduced=True)
# print ''

# level = 4
# dilation_size = 6
# seq_length = calc_seq_length(kernel_size,dilation_size,level)
# channel_sizes = [nhid] * level
# print "****Setup 2****"
# print "nlevel: %d" % level
# print "dilation_size: %d" % dilation_size
# print "kernel_size: %d" % kernel_size
# print "Sequence length: %d" % seq_length
# model = TCN(input_channels, n_classes, channel_sizes, dilation_size=dilation_size,kernel_size=kernel_size, dropout=dropout)
# s,sp = summary(model,input_size=(input_channels,seq_length),batch_size=batch_size,print_reduced=True)
# print ''

# level = 8
# dilation_size = 6
# seq_length = calc_seq_length(kernel_size,dilation_size,level)
# channel_sizes = [nhid] * level
# print "****Setup 2****"
# print "nlevel: %d" % level
# print "dilation_size: %d" % dilation_size
# print "kernel_size: %d" % kernel_size
# print "Sequence length: %d" % seq_length
# model = TCN(input_channels, n_classes, channel_sizes, dilation_size=dilation_size,kernel_size=kernel_size, dropout=dropout)
# s,sp = summary(model,input_size=(input_channels,seq_length),batch_size=batch_size,print_reduced=True)
# print ''
