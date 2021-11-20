import os
import tqdm

import torch
import numpy as np
import torch.nn as nn
import kornia
import time
import shutil

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from torch.nn.utils import clip_grad_norm_
from progress.bar import Bar


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 args
                 ):
        self.cfg = cfg
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.cuda()
        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model,
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        if cfg['sync_bn'] == True:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        ## DDP
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.args.local_rank], find_unused_parameters=True)

        #self.model = torch.nn.DataParallel(self.model).cuda()



    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step(self.epoch)


            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs('checkpoints', exist_ok=True)
                ckpt_name = os.path.join('checkpoints', 'checkpoint_epoch_%d' % self.epoch)

                if self.args.local_rank == 0:
                    save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

            if (self.epoch % self.cfg['eval_frequency']) == 0:
                if self.args.local_rank == 0:
                    self.inference()

            elif (self.max_epoch >self.cfg['max_epoch']-10):
                if self.args.local_rank == 0:
                    self.inference()

            progress_bar.update()

        return None


    def train_one_epoch(self):
        self.model.train()
        loss_stats = ['loss_rpn', 'loss_depth', 'rpn_loss_cls', 'rpn_loss', 'balancer_loss', 'fg_loss', 'bg_loss', 'ddn_loss']
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in loss_stats}
        num_iters = len(self.train_loader)
        bar = Bar('{}/{}'.format("3D", "CaDDN"), max=num_iters)
        end = time.time()

        #progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, inputs in enumerate(self.train_loader):
            for key, val in inputs.items():
                if not isinstance(val, np.ndarray):
                    continue
                if key in ['frame_id', 'metadata', 'calib']:
                    continue
                if key in ['images']:
                    inputs[key] = kornia.image_to_tensor(val).float().cuda()
                elif key in ['image_shape']:
                    inputs[key] = torch.from_numpy(val).int().cuda()
                else:
                    inputs[key] = torch.from_numpy(val).float().cuda()

            # train one batch
            self.optimizer.zero_grad()
            ret_dict, tb_dict  = self.model(inputs)

            #print(tb_dict)
            loss = ret_dict['loss'].mean()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                self.epoch, batch_idx, num_iters, phase="train",
                total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    tb_dict[l], inputs['images'].shape[0])
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            bar.next()
        bar.finish()

    def inference(self):
        # torch.set_grad_enabled(False)
        self.model.eval()
        rgb_results = {}
        dataset = self.test_loader.dataset
        class_names = dataset.class_name

        output_path = self.cfg['output_path']

        if os.path.exists(output_path):
            shutil.rmtree(output_path, True)
        os.makedirs(output_path, exist_ok=True)
        with torch.no_grad():
            progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
            for batch_idx, inputs in enumerate(self.test_loader):
                # load evaluation data and move data to GPU.
                for key, val in inputs.items():
                    if not isinstance(val, np.ndarray):
                        continue
                    if key in ['frame_id', 'metadata', 'calib']:
                        continue
                    if key in ['images']:
                        inputs[key] = kornia.image_to_tensor(val).float().cuda()
                    elif key in ['image_shape']:
                        inputs[key] = torch.from_numpy(val).int().cuda()
                    else:
                        inputs[key] = torch.from_numpy(val).float().cuda()

                pred_dicts, ret_dict = self.model(inputs, False)

                _ = dataset.generate_prediction_dicts(
                    inputs, pred_dicts, class_names,
                    output_path=output_path
                )

                progress_bar.update()

        progress_bar.close()

        self.test_loader.dataset.eval(results_dir=output_path, logger=self.logger)



