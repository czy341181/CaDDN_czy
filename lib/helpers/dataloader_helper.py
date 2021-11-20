import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

from torch.utils.data import DistributedSampler as _DistributedSampler

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(cfg, num_gpus, workers=4):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split='train', cfg=cfg)
        test_set = KITTI_Dataset(split='val', cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    # train_loader = DataLoader(dataset=train_set,
    #                           batch_size=cfg['batch_size'],
    #                           num_workers=workers,
    #                           worker_init_fn=my_worker_init_fn,
    #                           shuffle=True,
    #                           pin_memory=False,
    #                           collate_fn=train_set.collate_batch,
    #                           drop_last=True)
    # test_loader = DataLoader(dataset=test_set,
    #                          batch_size=cfg['batch_size'],
    #                          num_workers=workers,
    #                          worker_init_fn=my_worker_init_fn,
    #                          shuffle=False,
    #                          pin_memory=False,
    #                          collate_fn=test_set.collate_batch,
    #                          drop_last=False)

    batch_size = cfg['batch_size']//num_gpus

    # prepare dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=False,   #sample is specified
                              pin_memory=False,
                              collate_fn=train_set.collate_batch,
                              drop_last=True,
                              sampler=sampler)

    rank, world_size = get_dist_info()
    sampler = DistributedSampler(test_set, world_size, rank, shuffle=False)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             collate_fn=test_set.collate_batch,
                             drop_last=False,
                             sampler=sampler)

    return train_loader, test_loader
