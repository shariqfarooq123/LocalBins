import torch
import torch.nn as nn
import torch.cuda.amp as amp
from .base_trainer import BaseTrainer, TYPE_RGB, TYPE_DEPTH
# from .lr_finder_trainer_base import LRRTBaseTrainer as BaseTrainer
# from .base_profiler_trainer import BaseProfilerTrainer as BaseTrainer
from trainers.loss import SILogLoss, MultiScaleBinsLossV5
from utils.misc import compute_metrics
from data.types import DataLoaderTypes
from utils.bbox_utils import RandomBBoxQueries




class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader, test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss()
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        # self.chamfer_loss = MultiScaleGlobalBinsLoss()

        self.chamfer_loss = MultiScaleBinsLossV5(config)

        # self.d_pert_loss = DeviationPerturbationLoss(device=device)
        # self.v_pert_loss = VariancePerturbationLoss(device=device)


        # self.window_sizes = model.module.window_sizes if config.multigpu else model.window_sizes
        # self.window_sizes = [3, 7, 15, 31, 63]
        self.window_sizes = list(map(int, self.config.local_win_sizes.split(",")))

    def get_temperature(self):
        t = self.step * ((self.config.final_temp - self.config.init_temp) / (self.total_iters)) + self.config.init_temp
        # print("temperature", t)
        return t


    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of frames of images and depth as input
        batch["images"].shape : batch_size, n_frames, c, h, w
        batch["depths"].shape : batch_size, n_frames, 1, h, w
        """
        if self.config.dataloader_type == DataLoaderTypes.TUPLE:
            images, depths_gt = batch
            depths_gt = depths_gt.squeeze().unsqueeze(1)
        elif self.config.dataloader_type == DataLoaderTypes.DICT_MONO:
            images, depths_gt = batch['image'].to(self.device), batch['depth'].to(self.device)
        else:
            images, depths_gt = batch['images'].to(self.device), batch['depths'].to(self.device)

        

        b, c, h, w = images.size()
        bbox_queries = RandomBBoxQueries(b, h, w, self.window_sizes, N=self.config.n_bbox_queries).to(self.device)

        mask = torch.logical_and(depths_gt > self.config.min_depth, depths_gt < self.config.max_depth)
        t = self.get_temperature()
        losses = []
        # loss = 0
        with amp.autocast(enabled=self.config.use_amp):
            # centers, pred_heights, pred_depths = self.model(images, bbox_queries.normalized)
            response, pred_heights, pred_depths = self.model(images, bbox_queries.to(torch.float).absolute, t=t)

            l_si, pred = self.silog_loss(pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True, return_interpolated=True)
            loss = self.config.w_si * l_si

            if self.config.w_chamfer > 0:
                l_chamfer = self.chamfer_loss(response, bbox_queries, depths_gt)  # this is the multi-scale bins loss
                if l_chamfer is None:
                    l_chamfer = torch.Tensor([0])
                    orig_chamfer = torch.nan
                else:
                    loss = loss + self.config.w_chamfer * l_chamfer
                    # losses.append(self.config.w_chamfer * l_chamfer)
                    orig_chamfer = l_chamfer
            else:
                orig_chamfer = 0


            # loss = self.config.w_si * l_si \
            #     + self.config.w_chamfer * l_chamfer \
            #         + self.config.w_hmse * l_hmse
            # loss = sum(losses)
        # print(self.step, "loss: ", loss.item())
        self.scaler.scale(loss).backward()
        # print("Backward done", self.step)
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
        # print("End",self.step)

        return {f"{self.silog_loss.name}": l_si,
                f"{self.chamfer_loss.name}": orig_chamfer,
                }

    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)

        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        # images, depths_gt = map(lambda x: x.view(-1, *x.shape[-3:]), [images, depths_gt])
        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        with amp.autocast(enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
            pred_depths = m(images, t=self.get_temperature())[-1]
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)
        
        mask = torch.logical_and(depths_gt > self.config.min_depth, depths_gt < self.config.max_depth)
        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.silog_loss(pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)


        metrics = compute_metrics(depths_gt, pred_depths, **self.config)
        losses = {f"{self.silog_loss.name}": l_depth.item()}

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input":images[0]}, depth={"GT":depths_gt[0], "PredictedMono":pred_depths[0]}, prefix="Test")

        return metrics, losses


    

    



