import torch
import numpy as np
import logging
import random

import torch.multiprocessing as mp
import torch.distributed as dist

def create_logger(log_file, rank=0):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
                        format=log_format,
                        filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# def init_dist_pytorch():
#     if mp.get_start_method(allow_none=True) is None:
#         mp.set_start_method('spawn')
#     #tcp_port = 19000
#     num_gpus = torch.cuda.device_count()
#
#     dist.init_process_group(backend='nccl', init_method='env://')
#     # dist.init_process_group(
#     #     backend=backend,
#     #     init_method='tcp://127.0.0.1:%d' % tcp_port,
#     #     rank=local_rank,
#     #     world_size=num_gpus
#     # )
#     rank = dist.get_rank()
#     torch.cuda.set_device(rank)
#     return num_gpus, rank


def init_dist_pytorch(args):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    #tcp_port = 19000
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend='nccl', init_method='env://')
    # dist.init_process_group(
    #     backend=backend,
    #     init_method='tcp://127.0.0.1:%d' % tcp_port,
    #     rank=local_rank,
    #     world_size=num_gpus
    # )
    return num_gpus