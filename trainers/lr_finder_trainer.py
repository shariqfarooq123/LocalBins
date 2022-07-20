import torch
import torch.nn as nn
import torch.cuda.amp as amp
# from .base_trainer import BaseTrainer, TYPE_RGB, TYPE_DEPTH
# from .base_profiler_trainer import BaseProfilerTrainer as BaseTrainer
from .lr_finder_trainer_base import LRRTBaseTrainer as BaseTrainer
from trainers.loss import SILogLoss
from utils.misc import compute_metrics
import torch.optim as optim




            



class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader, test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss()
        self.scaler = amp.GradScaler()
        # self.chamfer_loss = BinsChamferLoss()



    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of frames of images and depth as input
        batch["images"].shape : batch_size, n_frames, c, h, w
        batch["depths"].shape : batch_size, n_frames, 1, h, w
        """

        images, depths_gt = batch
        

        b, c, h, w = images.size()
        mask = torch.logical_and(depths_gt > self.config.min_depth, depths_gt < self.config.max_depth)

        with amp.autocast():
            pred_depths = self.model(images)

            l_si, pred = self.silog_loss(pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True, return_interpolated=True)

            loss = l_si

        self.scaler.scale(loss).backward()
        # loss.backward()
        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

        self.scaler.step(self.optimizer)
        # self.optimizer.step()

        if self.should_log and self.step > 1 and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred[0]}, prefix="Train")


        self.scaler.update()
        self.optimizer.zero_grad()

        return {f"{self.silog_loss.name}": l_si
                }

    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)

        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        # images, depths_gt = map(lambda x: x.view(-1, *x.shape[-3:]), [images, depths_gt])
        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        pred_depths = self.model(images)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)
        
        mask = torch.logical_and(depths_gt > self.config.min_depth, depths_gt < self.config.max_depth)
        l_depth = self.silog_loss(pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)

        metrics = compute_metrics(depths_gt, pred_depths, **self.config)
        losses = {f"{self.silog_loss.name}": l_depth.item()}

        if val_step == 1:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input":images[0]}, depth={"GT":depths_gt[0], "PredictedMono":pred_depths[0]}, prefix="Test")

        return metrics, losses


    

    



