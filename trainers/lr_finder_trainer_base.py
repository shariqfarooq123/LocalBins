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


class LRRTScheduler(object):
    def __init__(self, optimizer, init_lr=1e-4, max_lr=10, num_iter=100, param_group_factors=(0.1,1)):
        self.optimizer = optimizer
        self._step = 0
        self.num_iter = num_iter
        self.param_group_factors = param_group_factors
        self.init_lr = init_lr
        self.max_lr = max_lr

    def step(self):
        lr = self.init_lr * ((self.max_lr / self.init_lr) ** (self._step / self.num_iter))
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.param_group_factors[i] * lr

        self._step += 1

        if self._step > self.num_iter:
            raise RuntimeError("NOT AN ERROR. LR test finished")

        return lr


class LRRTBaseTrainer:
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
        return LRRTScheduler(self.optimizer)
        

    def train_on_batch(self, batch, train_step):
        raise NotImplementedError

    def validate_on_batch(self, batch, val_step):
        raise NotImplementedError

    def train(self):
        print(f"Training {self.config.name}")
        
        run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{self.config.bs}-tep{self.config.epochs}-lr{self.config.lr}-wd{self.config.wd}-{uuid.uuid4()}"
        self.config.run_id = run_id
        self.config.experiment_id = f"{self.config.name}{self.config.version_name}_{run_id}"
        self.should_write = ((not self.config.distributed) or self.config.rank == 0)
        self.should_log = self.should_write # and logging
        if self.should_log:
            tags = self.config.tags.split(',') if self.config.tags != '' else None
            wandb.init(project=self.config.project, name=self.config.experiment_id, config=flatten(self.config), dir=self.config.root, tags=tags, notes=self.config.notes)
        
        self.model.train()
        self.iters_per_epoch = len(self.train_loader)
        self.step = 0
        best_loss = np.inf
        validate_every = int(self.config.validate_every * self.iters_per_epoch)
        for epoch in range(self.config.epochs):
            self.epoch = epoch
             ################################# Train loop ##########################################################
            if self.should_log: wandb.log({"Epoch": epoch}, step=self.step)

            for i, batch in tqdm(enumerate(self.train_loader), desc=f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train",
                                total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader):

                losses = self.train_on_batch(batch, i)
                curr_lr = self.scheduler.step()

                if self.should_log and self.step % 1 == 0:
                    wandb.log({"LearningRate": curr_lr}, step=self.step)
                    wandb.log({f"Train/{name}": loss.item() for name, loss in losses.items()}, step=self.step)

                self.step += 1

                ########################################################################################################
               

                if self.test_loader:
                    if self.should_write and (self.step % validate_every) == 0:
                        self.model.eval()
                        # self.save_checkpoint(f"{self.config.experiment_id}_latest.pt")

                        ################################# Validation loop ##################################################
                        metrics, test_losses = self.validate()
                        # print("Validated: {}".format(metrics))
                        if self.should_log:
                            wandb.log({f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)
                            
                            wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=self.step)

                            # if (metrics['abs_rel'] < best_loss) and self.should_write:
                            #     self.save_checkpoint(f"{self.config.experiment_id}_best.pt")
                            #     best_loss = metrics['abs_rel']

                        self.model.train()
                #################################################################################################

        # Save / validate at the end
        self.step += 1  # log as final point
        self.model.eval()
        # self.save_checkpoint(f"{self.config.experiment_id}_latest.pt")
        if self.test_loader:
            if self.should_write:
                ################################# Validation loop ##################################################
                metrics, test_losses = self.validate()
                # print("Validated: {}".format(metrics))
                if self.should_log:
                    wandb.log({f"Test/{name}": tloss for name, tloss in test_losses.items()}, step=self.step)
                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=self.step)

                    if (metrics['abs_rel'] < best_loss) and self.should_write:
                        self.save_checkpoint(f"{self.config.experiment_id}_best.pt")
                        best_loss = metrics['abs_rel']

        self.model.train()

    def validate(self):
        with torch.no_grad():
            losses_avg = RunningAverageDict()
            metrics_avg = RunningAverageDict()
            for i, batch in tqdm(enumerate(self.test_loader), desc=f"Epoch: {self.epoch + 1}/{self.config.epochs}. Loop: Validation", total=len(self.test_loader)) if is_rank_zero(
                    self.config) else self.test_loader:
                metrics, losses = self.validate_on_batch(batch, val_step=i)
                
                if losses: losses_avg.update(losses)
                if metrics: metrics_avg.update(metrics)

            return metrics_avg.get_value(), losses_avg.get_value()

    def save_checkpoint(self, filename):
        root = self.config.save_dir
        if not os.path.isdir(root):
            os.makedirs(root)

        fpath = os.path.join(root, filename)
        m = self.model.module if self.config.multigpu else self.model
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch
            }
            , fpath)

    def log_images(self, rgb: Dict[str,list]={}, depth: Dict[str,list]={}, scalar_field: Dict[str,list]={}, prefix=""):
        depth = {k: colorize(v, vmin=self.config.min_depth, vmax=self.config.max_depth) for k,v in depth.items()}
        scalar_field = {k: colorize(v, vmin=None, vmax=None, cmap='jet') for k,v in scalar_field.items()}
        images = {**rgb, **depth, **scalar_field}
        wimages = {prefix+"Predictions": [wandb.Image(v, caption=k) for k,v in images.items()]}
        wandb.log(wimages, step=self.step)

    def log_line_plot(self, data):
        plt.plot(data)
        plt.ylabel("Scale factors")
        wandb.log({"Scale factors": wandb.Image(plt)}, step=self.step)
        plt.close()

    def log_bar_plot(self, title, labels, values):
        data = [[label, val] for (label,val) in zip(labels, values)]
        table = wandb.Table(data=data, columns=["label", "value"])
        wandb.log({title: wandb.plot.bar(table, "label", "value", title=title)}, step=self.step)

    # def log_image_grid(self, images, caption="", prefix="", size=(416,544), **kwargs):

    #     imgs = []
    #     for imtype, im in images:
    #         if isinstance(im, torch.Tensor):
    #             im = im.detach().cpu().numpy()
    #         if imtype == TYPE_DEPTH:
    #             im = colorize(im, vmin=self.config.min_depth, vmax=self.config.max_depth)
    #             im = im.transpose(2, 0, 1)
            
    #         im = torch.Tensor(im)
    #         if imtype == TYPE_RGB:
    #             im = denormalize(im).squeeze()
    #         if im.shape[-2:] != size:
    #             im = nn.functional.interpolate(im.unsqueeze(0), size=size, mode='bilinear', align_corners=True).squeeze()

    #         imgs.append(im.squeeze())

    #     grid = F.to_pil_image(make_grid(imgs, **kwargs))
    #     wimages = {prefix+"Predictions": wandb.Image(grid, caption=caption)}
    #     wandb.log(wimages, step=self.step)











    
