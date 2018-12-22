#!/usr/bin python

#I want to verify my equations for the receptive field. The one tricky part is that by default,
#each residual block has 2 convolutions, and I'm not sure if this effectively increases the receptive
#field also. 
#To test, I will create a model with kernel size, dilation, and hidden layers, and with the sequence
#length I expect, I will put a sequence of that length, but with a single 1 at the end. If I'm correct,
#only the last point will have a value (everything else should be 0's)
from model import TCN
import numpy as np
from torchsummary import summary
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt; plt.ion()

from scipy.optimize import fsolve
def funcLevel(x,*args):
     S,k,d = args
     return 1 + (k-1)*(d**x-1.)/(d-1.) - S

root = './data/mnist'
batch_size = 1
n_classes = 2
input_channels = 1
kernel_size = 6
nhid = 1
dropout = 0.05

def calc_seq_length(kernel_size,dilation_sizes,nlevel):
     """Assumes kernel_size and dilation_size scalar, dilation_size exponential increase"""
     if np.isscalar(dilation_sizes): dilation_sizes = dilation_sizes**np.arange(nlevel)
     return 1 + 2*(kernel_size-1)*np.sum(dilation_sizes)

sequence_length = 10000
d = 4
# total_sizes = []
# seq_lengths = []
#levelfrac = fsolve(funcLevel,30,args=(sequence_length,kernel_size,d))
#level = int(np.ceil(levelfrac))

level = 5
channel_sizes = [nhid] * level
#seq_length = calc_seq_length(kernel_size,d,level)
#last_dilation = int(np.ceil((sequence_length - calc_seq_length(kernel_size,d,level-1))/(kernel_size-1)))
#dilation_sizes = (d**np.arange(level-1)).tolist() + [last_dilation]
dilation_sizes = (d**np.arange(level)).tolist()
seq_length = calc_seq_length(kernel_size,dilation_sizes,level)
#print d,seq_length,level

#continue
model = TCN(input_channels, n_classes, channel_sizes, dilation_size=dilation_sizes,kernel_size=kernel_size, dropout=0.0)
s,sp = summary(model,input_size=(input_channels,seq_length),batch_size=batch_size,noprint=True)

data = np.zeros((batch_size,input_channels,sequence_length))
data = Variable(torch.from_numpy(data).type(torch.FloatTensor))
data2 = data.clone()
data2[0,0,0] = 10.
out = model(data)
out2 = model(data2)
plt.figure()
plt.plot(out[0,0,:].detach().numpy())
plt.plot(out2[0,0,:].detach().numpy())

seq_lengthAct = np.max(np.where(out2[0,0,:].detach().numpy()>1e-6)) + 1
print "k,d,level,seq_length (Theory), seq_length (Actual)"
print kernel_size, d, level, seq_length, seq_lengthAct
