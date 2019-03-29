import math
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np

class DistributedStratifiedSampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of the dataset, 
       and ensures balanced classes in each batch

    See DistributedSampler docs for more details

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        stratify (optional): Labels to balance among batches
    """

    def __init__(self, dataset, num_replicas=None, rank=None, stratify=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        if stratify is not None:
            self.stratify = stratify
            self.pos_stratify = np.where(stratify==1)[0]
            self.neg_stratify = np.where(stratify==0)[0]
            self.Npos = int(sum(stratify))
            self.Nneg = int(stratify.size - sum(stratify))
            self.pos_num_samples = int(math.ceil(self.Npos * 1.0 / self.num_replicas))
            self.neg_num_samples = int(math.ceil(self.Nneg * 1.0 / self.num_replicas))
            self.pos_total_size = self.pos_num_samples * self.num_replicas
            self.neg_total_size = self.neg_num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.stratify is not None:
            pos_indices = torch.randperm(self.Npos, generator=g).tolist()
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
