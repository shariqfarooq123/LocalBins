import os
from typing import Dict
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import wandb
from datetime import datetime as dt
import uuid
import torch.optim as optim
from utils.misc import RunningAverage, RunningAverageDict, colorize, denormalize
from utils.config import flatten
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

TYPE_RGB = 'rgb'
TYPE_DEPTH = 'depth'


def is_rank_zero(args):
    return args.rank == 0

class BaseProfilerTrainer:
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        self.config = config
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

    
    def init_optimizer(self):
        if self.config.same_lr:
            print("Using same LR")
            params = self.model.parameters()
            # self._lr = self.config.lr
        else:
            print("Using diff LR")
            m = self.model.module if self.config.multigpu else self.model
            lr = self.config.lr
            params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},  
                  {"params": m.get_10x_lr_params(), "lr": lr}]

            # self._lr = [lr / 10, lr]

        return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.wd)

    def init_scheduler(self):
        lrs = [l['lr'] for l in self.optimizer.param_groups]
        return optim.lr_scheduler.OneCycleLR(self.optimizer, lrs, epochs=self.config.epochs, steps_per_epoch=len(self.train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, div_factor=self.config.div_factor, final_div_factor=self.config.final_div_factor
, pct_start=self.config.pct_start                                              )

    def train_on_batch(self, batch, train_step):
        raise NotImplementedError

    def validate_on_batch(self, batch, val_step):
        raise NotImplementedError

    def train(self):
        print(f"Training {self.config.name}")

        self.should_log = self.should_write = False # Turn off logging

        self.model.train()
        self.iters_per_epoch = len(self.train_loader)
        self.step = 0
        prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/profile_{self.config.name}'),
                record_shapes=True,
                with_stack=True)
        prof.start()
        ################################# Train loop ##########################################################

        for i, batch in tqdm(enumerate(self.train_loader), desc=f"Profiling",
                            total=(1 + 1 + 3) * 2) if is_rank_zero(self.config) else enumerate(self.train_loader):
            if self.step >= (1 + 1 + 3) * 2:
                break
            losses = self.train_on_batch(batch, i)
            self.scheduler.step()
            self.step += 1

            prof.step()
        prof.stop()