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
        undersample (optional): Fraction of neg/pos samples desired (e.g. 1.0 for equal; 0.5 for 1/3 neg, 2/3 pos, etc.)   
        distributed (optional): Stratified DistributedSampler
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, stratify=None, undersample=None,
                distributed=False, num_replicas=None, rank=None):
        self.stratify = stratify
        self.undersample = undersample
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
            self.pos_num_samples = int(math.ceil(self.Npos * 1.0 / self.num_replicas))
            if self.undersample is not None:
                self.neg_num_samples = int(self.undersample*self.pos_num_samples)
            else:
                self.neg_num_samples = int(math.ceil(self.Nneg * 1.0 / self.num_replicas))
            self.num_samples = self.pos_num_samples + self.neg_num_samples
            self.pos_total_size = self.pos_num_samples * self.num_replicas
            self.neg_total_size = self.neg_num_samples * self.num_replicas
            if self.undersample is not None:
                g = torch.Generator()
                g.manual_seed(0)
                neg_indices = torch.randperm(self.Nneg, generator=g)
                self.neg_num_samples = int(self.undersample*self.pos_num_samples)
                self.neg_indices_init = neg_indices[:self.neg_total_size]
                self.neg_used_indices = self.neg_stratify[self.neg_indices_init]
            else:
                self.neg_used_indices = self.neg_stratify
            #global indices used (fixed over epochs), for convenience in reading out
            self.pos_used_indices = self.pos_stratify

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.stratify is not None:
            pos_indices = torch.randperm(self.Npos, generator=g).tolist()
            if self.undersample is not None:
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
                indices.append(pos_indices[i])
                indices.extend(neg_indices[j:j+nfact])

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
