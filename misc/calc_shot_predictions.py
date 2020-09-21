#!/usr/bin python

#just plot the predictions, for visual investigation


import numpy as np
import h5py
import matplotlib.pyplot as plt; plt.ion()

import optuna


def load_data(preds_file,setname='val'):    
    '''Preload all data'''
    #hack since portal cant have torch
    class tmp(): pass
    args = tmp(); args.data_step=10 
    args.nrecept = 30017

    f = h5py.File(preds_file,'r')
    shots = f['shots_'+setname][...]
    start_idx = (f['start_idx_'+setname][...]/args.data_step).astype(int)+args.nrecept #have to add nrecept, since otherwise reliant on rampup
    stop_idx = (f['stop_idx_'+setname][...]/args.data_step).astype(int)
    disrupt_idx = (f['disrupt_idx_'+setname][...]/args.data_step).astype(int)
    disrupted = f['disrupted_'+setname][...]
    preds = []
    for i,shot in enumerate(shots):
        pred = f['preds/'+str(shot)][0,...] #first dimension superfluous
        istart_idx = int(start_idx[i])# + args.nrecept)
        istop_idx = int(stop_idx[i])
        preds += [pred[istart_idx:istop_idx+1]]
    f.close()
    
    return preds,shots,start_idx,stop_idx,disrupt_idx,disrupted,args


def create_alarms(pred,thresh_low,thresh_high):
    '''create alarms, only resetting after a high threshold drops back below low'''
    alarms = np.zeros(pred.shape)

    #create boolean arrays
    glow = pred <= thresh_low
    ghigh = pred >= thresh_high
    gmid = ~(glow | ghigh)

    alarms[ghigh] = 1

    inds = np.where(gmid)[0]
    startinds = inds[np.where(np.diff(inds)>1)[0]+1]
    stopinds = inds[np.where(np.diff(inds)>1)[0]]
    if inds.size>0:
        startinds = np.concatenate(([inds[0]],startinds))
        stopinds = np.concatenate((stopinds,[inds[-1]]))
    #loop through start and stop inds, using the last point
    for i,(istart,istop) in enumerate(zip(startinds,stopinds)):
        if istart==0:
            alarms[istart:istop+1] = 0
        else:
            alarms[istart:istop+1] = alarms[istart-1]
    return alarms


def alarm_hysteresis(alarms,Nalarm):
    '''Determine sequence index of first alarm sequence beyond Nalarm points long'''
    startinds = np.where(np.diff(alarms)==1)[0]
    if alarms[0]==1:
        startinds = np.concatenate(([0],startinds))
    if startinds.size==0: return -1000
    
    alarminds = np.where(alarms)[0]
    alarmseqlengths = np.diff(np.concatenate(([-1],np.where(np.diff(alarminds)>1)[0],[alarminds.size-1])))
    alarmseq = np.where(alarmseqlengths>Nalarm)[0]
    if alarmseq.size==0: return -1000
    firstseqind = startinds[alarmseq[0]]+Nalarm
    return firstseqind
   

    
class Objective(object):
    def __init__(self,preds_file,setname='val',discrete=True,metric='f1'):
        self.preds_file = preds_file
        self.discrete = discrete
        self.setname = setname
        self.metric = metric
        self.preds,self.shots,self.start_idx,self.stop_idx,self.disrupt_idx,self.disrupted,self.args = load_data(self.preds_file,setname=self.setname)
        self.dt = 0.001*self.args.data_step
        self.Tmin = 30 #ms, minimum time before disruptin needed for mitigation
        self.Nmin = self.Tmin/self.dt

    def calc_shot_confusion(self,thresh_low,thresh_high,Talarm,Tclass,plot=False):        
        Nalarm = Talarm/self.dt
        Nclass = Tclass/self.dt
        TP = 1e-10; FP = 1e-10; TN = 1e-10; FN = 1e-10
        self.disruptioninds = np.empty((len(self.preds),))
        for i,shot in enumerate(self.shots):
            alarms = create_alarms(self.preds[i],thresh_low,thresh_high)
            self.disruptioninds[i] = alarm_hysteresis(alarms,Nalarm)
            alarm_on = self.disruptioninds[i]>=0
            if self.disrupted[i]:
                if ((alarm_on) & (self.disruptioninds[i]<=(self.stop_idx[i]-self.Nmin-self.start_idx[i])) & (self.disruptioninds[i]>=(self.stop_idx[i] - self.start_idx[i] - Nclass))):
                    TP += 1
                else:
                    FN += 1
            else:
                if alarm_on:
                    FP += 1
                else:
                    TN += 1
            if plot:
                self.plot_data(i,alarms=alarms,disruptionind=self.disruptioninds[i])
                plt.draw()
                input("Hit enter to continue")

        precision = TP/(TP+FP)+1e-10 
        recall = TP/(TP+FN)+1e-10        
        f1 = 2./(1./precision + 1./recall)
        f2 = (1+2**2)*precision*recall/(2**2*precision + recall)

        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)

        #try to minimize distance from perfect TPR=1 and FPR=0
        tpr_fpr_dist = np.sqrt((TPR-1)**2. + (FPR)**2.)

        return TP,FP,TN,FN,precision,recall,f1,f2,tpr_fpr_dist

    def plot_data(self,index,alarms=None,disruptionind=-1):
        plt.clf()
        t = -50 + np.arange(self.start_idx[index],self.stop_idx[index]+1)*self.dt
        plt.plot(t,self.preds[index])
        if alarms is not None:
            plt.plot(t,alarms,'g',linewidth=0.25)
        if self.disrupted[index]:
            plt.plot(2*[t[int(self.stop_idx[index]-self.start_idx[index]-self.Nmin)]],[0,1],'r--')
            plt.plot(2*[t[int(self.stop_idx[index]-self.start_idx[index])]],[0,1],'r-')
        if disruptionind>0:
            plt.plot(2*[t[int(disruptionind)]],[0,1],'k--')
        if ((self.disrupted[index]) ^ (disruptionind>-1)):
            color = 'red'
        else:
            color = 'green'
        plt.title("Shot: %d, Disrupted: %s, disruptionind: %d" % (self.shots[index],str(self.disrupted[index]),disruptionind),color=color)
        plt.ylim([-0.1,1.1])

    def warning_times(self,plot=False):
        self.Ndisrupt = np.sum(self.disrupted)
        #stop_idx w.r.t entire sequence; ` disruptioninds w.r.t to shortened sequence
        Ndiff = (self.stop_idx-self.start_idx)-self.disruptioninds
        Ndiff = Ndiff[(self.disrupted) & (self.disruptioninds>0)]
        self.talarm_before_disrupt = Ndiff*self.dt
        self.talarm_before_disrupt.sort()
        if plot:
            x = self.talarm_before_disrupt.copy()/1000 #convert to s
            y = np.arange(self.talarm_before_disrupt.size)[::-1]/self.Ndisrupt*100
            #if no points are less than 1e-3, add flat point to make plot continue to 1e-3
            if x.min()>1e-3:
                x = np.concatenate(([1e-3],x))
                y = np.concatenate(([y[0]],y))
            plt.figure()
            plt.semilogx(x,y)
            plt.plot([0.03,0.03],[0,100],'k--')
            plt.xlabel('Disruption time - Alarm time [s]')
            plt.ylabel('Accumulated fraction of \ndetected disruptions (%)')
            plt.tight_layout()
            plt.grid()
            plt.xlim([1e-3,10])
            plt.gca().set_yticks(np.linspace(0,100,11))
            plt.ylim([0,100])
            plt.grid(which='minor')        
            
            
    def __call__(self,trial):
        if self.discrete:
            #thresh_low = trial.suggest_discrete_uniform('thresh_low',0.05,0.5,0.05)
            #thresh_low = trial.suggest_discrete_uniform('thresh_low',0.05,0.2,0.05)
            #thresh_high = trial.suggest_discrete_uniform('thresh_high',0.5,0.95,0.05)
            thresh_high = trial.suggest_discrete_uniform('thresh_high',0.8,0.99,0.01)
            thresh_low = trial.suggest_discrete_uniform('thresh_low',0.01,thresh_high,0.01)
            #Talarm = trial.suggest_discrete_uniform('Talarm',5,1000,5)
            Talarm = trial.suggest_discrete_uniform('Talarm',0,100,1)
            #Tclass = trial.suggest_discrete_uniform('Tclass',300,5000,100)
            Tclass = 10000
        else:
            thresh_low = trial.suggest_uniform('thresh_low',0.0,0.5)
            thresh_high = trial.suggest_uniform('thresh_high',0.5,1.0)
            Talarm = trial.suggest_uniform('Talarm',5,1000)

        _,_,_,_,_,_,f1,_,tpr_fpr_rate = self.calc_shot_confusion(thresh_low,thresh_high,Talarm,Tclass)

        if 'f1' in self.metric.lower():
            metric=f1
        else:
            metric=tpr_fpr_rate

        return metric
    
    
if __name__=='__main__':
    predsTest_file = 'test_predictions_4509197.h5'
    #predsTest_file = 'test_predictions_4532347.h5'
    objTest = Objective(predsTest_file,setname='test')

    preds_file = 'validation_predictions_4509197.h5'
    #preds_file = 'validation_predictions_4532347.h5'
    obj = Objective(preds_file,setname='val',metric='tpr_fpr_dist')

    if 'f1' in obj.metric.lower():
        study = optuna.create_study(direction='maximize')
    else:
        study = optuna.create_study(direction='minimize')
        
    study.optimize(obj,n_trials=10000)
    print(study.best_params) 
