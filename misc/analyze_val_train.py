#!/usr/bin python

import glob
import os
import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import re
import sys

#this is for the newer log files, which gather statistics over distributed nodes,
#and generally is done every epoch instead of every iteration

def read_losses(filename):
    epochs = []; losses = []; iterations = []; steps = []; times = []; lrs = []
    val_index = []; val_losses = []; val_acc = []; val_tpr = []; val_fpr = [];  val_f1 = []; val_times = []; thresholds = []
    lr = 0
    with open(filename) as f:
        for line in f:
            if "-lr=" in line:
                lr = float(re.findall("\d+\.\d+",line)[0])
            if "Train Epoch" in line:
                epochs += [float(line.split('Epoch:')[1].split()[0])]
                losses += [float(line.split('Loss:')[1].split()[0])]
                iterations += [float(line.split('Iteration:')[1].split()[0])]
                steps += [float(line.split('Steps:')[1].split()[0])]
                times += [float(line.split('Time:')[1].split()[0])]
                if "LR" in line:
                    lrs += [float(line.split('LR:')[1].split()[0])]
            if "Validation set" in line:
                if ',' in line:
                    line = line.replace(',','') #older log files, newer has \t
                    val_acc += [float(line.split('Accuracy:')[1].split()[1][1:3])/100.]
                else:
                    val_acc += [float(line.split('Accuracy:')[1].split()[0])]
                val_index += [len(epochs)-1]
                val_losses += [float(line.split('Average loss:')[1].split()[0])]
                if 'TPR' in line: #TPR introduced later, this for backwards compatability
                    val_tpr += [float(line.split('TPR:')[1].split()[0])]
                    val_fpr += [float(line.split('FPR:')[1].split()[0])]
                else:
                    val_tpr += [0]
                    val_fpr += [0]
                val_f1 += [float(line.split('F1:')[1].split()[0])]
                if 'Threshold' in line:
                    thresholds += [float(line.split('Threshold:')[1].split()[0])]
                times += [float(line.split('Time:')[1].split()[0])]
    epochs = np.array(epochs); losses = np.array(losses); iterations = np.array(iterations)
    steps = np.array(steps); times = np.array(times); lrs = np.array(lrs)
    val_epochs = np.array([epochs[i] for i in val_index])
    val_iterations = np.array([iterations[i] for i in val_index])
    val_losses = np.array(val_losses); val_acc = np.array(val_acc); val_f1 = np.array(val_f1)
    return lr,epochs,losses, iterations, steps, times, lrs, \
           val_epochs, val_iterations, val_losses, val_acc, val_tpr, val_fpr, val_f1, thresholds

if __name__=="__main__":
    files = glob.glob('slurm*out')
    files.sort(key=os.path.getmtime)
    if sys.argv[1] is not None:
        files = []
        strs = sys.argv[1:]
        for stri in strs:
            tmp = stri.split(',')
            for tmpi in tmp:
                files += glob.glob(tmpi)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i,f in enumerate(files):
        #if '615' in f: continue
        lr,epochs,losses, iterations, steps, times, lrs, val_epochs, val_iterations, val_losses, val_acc, val_tpr, val_fpr, val_f1, thresholds = read_losses(f)
        if i==0:
            plt.figure(1,figsize=[13,4.5]); plt.clf()
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132,sharex=ax1)
            ax3 = plt.subplot(133,sharex=ax1)
            plt.tight_layout()
        ax1.plot(val_epochs,val_losses,'-o',color=colors[i],alpha=0.25)
        ax1.plot(epochs,losses,label=str(lr)+','+f,color=colors[i])
        ax1.set_yscale('log')
        ax2.plot(val_epochs,val_acc,'-o',color=colors[i],)
        ax2.plot(val_epochs,val_f1,'-d',color=colors[i])
        ax3.plot(epochs,lrs,label=str(lr)+','+f,color=colors[i])
    plt.legend()

