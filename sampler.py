import math
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np

class StratifiedSampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of the dataset, 
       and ensures balanced classes in each batch (currently only binary classes)

    See DistributedSampler docs for more details

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        stratify (optional): Labels to balance among batches
        undersample (optional): Fraction of neg/pos samples desired, dropping neg samples (e.g. 1.0 for equal; 0.5 for 1/3 neg, 2/3 pos, etc.)   
        oversample (optional): Duplicate pos samples, match with equal neg samples. Currently only integer input. 
                               (e.g. if total N = Npos + Nneg, and Nneg/Npos = 10, oversample=2 will set the # positive samples as 2*Npos, and
                               # negative samples as 2*Npos (i.e. 5x smaller than available) )   
        distributed (optional): Stratified DistributedSampler
        num_replicas (optional): Number of processes participating in
            distributed training. DistributedSampler will autofill to world size if not specified. 
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, stratify=None, undersample=None, oversample=None,
                distributed=False, num_replicas=None, rank=None):
        self.stratify = stratify
        self.undersample = undersample
        self.oversample = oversample
        self.distributed = distributed
        if self.distributed:
            DistributedSampler.__init__(self,dataset, num_replicas=num_replicas, rank=rank)
        else:
            #TODO need to create ither variables defined in distriburedsampler when stratify off
            self.num_replicas = 1
            self.rank = 0
            self.epoch = 0
        if self.stratify is not None:
            self.pos_stratify = np.where(self.stratify==1)[0]
            self.neg_stratify = np.where(self.stratify==0)[0]
            self.Npos = int(sum(self.stratify))
            self.Nneg = int(self.stratify.size - sum(self.stratify))
            
            #number of local samples (pos/neg_num_samples), and total global size (pos/neg_total_size)
            #note pos/neg_total_size are similar to Npos/Nneg, but larger to be evenly divisble by num_replicas
            self.pos_num_samples = int(math.ceil(self.Npos * 1.0 / self.num_replicas))
            self.neg_num_samples = int(math.ceil(self.Nneg * 1.0 / self.num_replicas))
            if self.undersample is not None:
                self.neg_num_samples = int(self.undersample*self.pos_num_samples)
            elif self.oversample is not None:
                self.pos_num_samples = int(self.oversample*self.pos_num_samples)
                assert self.pos_num_samples <= self.neg_num_samples, 'ERROR: oversampling too high, more than available'
                self.neg_num_samples = self.pos_num_samples
            self.num_samples = self.pos_num_samples + self.neg_num_samples
            self.pos_total_size = self.pos_num_samples * self.num_replicas
            self.neg_total_size = self.neg_num_samples * self.num_replicas

            if self.undersample is not None:
                g = torch.Generator()
                g.manual_seed(0)
                neg_indices = torch.randperm(self.Nneg, generator=g)
                self.neg_indices_init = neg_indices[:self.neg_total_size]
                self.neg_used_indices = self.neg_stratify[self.neg_indices_init]
            elif self.oversample is not None:
                g = torch.Generator()
                g.manual_seed(0)
                neg_indices = torch.randperm(self.Nneg, generator=g)
                self.neg_indices_init = neg_indices[:self.neg_total_size]
                self.neg_used_indices = self.neg_stratify[self.neg_indices_init]
                #replicate indices of positive samples
                pos_indices = torch.cat(int(self.oversample)*[torch.randperm(self.Npos, generator=g)])
                self.pos_indices_init = pos_indices[torch.randperm(len(pos_indices),generator=g)]
                self.pos_used_indices = self.pos_stratify[self.pos_indices_init]
            else:
                #global indices used (fixed over epochs), for convenience in reading out
                self.neg_used_indices = self.neg_stratify
                self.pos_used_indices = self.pos_stratify

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.stratify is not None:
            if self.oversample is not None:
                indices = torch.randperm(len(self.pos_indices_init), generator=g)
                pos_indices = self.pos_indices_init[indices].tolist()
            else:
                pos_indices = torch.randperm(self.Npos, generator=g).tolist()
            if (self.undersample is not None) or (self.oversample is not None):
                indices = torch.randperm(len(self.neg_indices_init), generator=g)
                neg_indices = self.neg_indices_init[indices].tolist()
            else:
                neg_indices = torch.randperm(self.Nneg, generator=g).tolist()


            # add extra samples to make it evenly divisible
            pos_indices += pos_indices[:(self.pos_total_size - len(pos_indices))]
            neg_indices += neg_indices[:(self.neg_total_size - len(neg_indices))]
            assert len(pos_indices) == self.pos_total_size
            assert len(neg_indices) == self.neg_total_size

            # subsample
            pos_indices = pos_indices[self.rank:self.pos_total_size:self.num_replicas]
            neg_indices = neg_indices[self.rank:self.neg_total_size:self.num_replicas]
            assert len(pos_indices) == self.pos_num_samples
            assert len(neg_indices) == self.neg_num_samples

            # pos/neg to global inds
            pos_indices = self.pos_stratify[pos_indices]
            neg_indices = self.neg_stratify[neg_indices]

            # interleave
            nfact = math.ceil(len(neg_indices)/len(pos_indices))
            indices = []
            for i,j in enumerate(range(0,len(neg_indices),nfact)):
                if self.rank % 2: #this ensures local batch_size=1 still gives balanced global sets
                    indices.append(pos_indices[i])
                    indices.extend(neg_indices[j:j+nfact])
                else:
                    indices.extend(neg_indices[j:j+nfact])
                    indices.append(pos_indices[i])

            return iter(indices)
        else:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

            return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


#example, experimental (not tested)
#from https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143
#class DistributedWeightedSampler(Sampler):
#    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
#        if num_replicas is None:
#            if not dist.is_available():
#                raise RuntimeError("Requires distributed package to be available")
#            num_replicas = dist.get_world_size()
#        if rank is None:
#            if not dist.is_available():
#                raise RuntimeError("Requires distributed package to be available")
#            rank = dist.get_rank()
#        self.dataset = dataset
#        self.num_replicas = num_replicas
#        self.rank = rank
#        self.epoch = 0
#        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
#        self.total_size = self.num_samples * self.num_replicas
#        self.replacement = replacement
#    
#    def calculate_weights(self, targets):
#        class_sample_count = torch.tensor(
#                [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
#        weight = 1. / class_sample_count.double()
#        samples_weight = torch.tensor([weight[t] for t in targets])
#        return samples_weight
#
#    def __iter__(self):
#        # deterministically shuffle based on epoch
#        g = torch.Generator()
#        g.manual_seed(self.epoch)
#        if self.shuffle:
#            indices = torch.randperm(len(self.dataset), generator=g).tolist()
#        else:
#            indices = list(range(len(self.dataset)))
#        # add extra samples to make it evenly divisible
#        indices += indices[:(self.total_size - len(indices))]
#        assert len(indices) == self.total_size
#        # subsample
#        indices = indices[self.rank:self.total_size:self.num_replicas]
#        assert len(indices) == self.num_samples
#        
#        # get targets (you can alternatively pass them in __init__, if this op is expensive)
#        targets = self.dataset.targets            
#        targets = targets[self.rank:self.total_size:self.num_replicas]
#        assert len(targets) == self.num_samples
#        weights = self.calculate_weights(targets)
#        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())
#    
#    def __len__(self):
#        return self.num_samples
#
#    def set_epoch(self, epoch):
#        self.epoch = epoch
