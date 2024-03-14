import time
import os
import argparse
import torch as T
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_logger, cmap_to_lbl, calculate_iou, calculate_iou_one_class
from typing import Dict
from tqdm import tqdm
from collections import deque

class Agent():
    def __init__(
        self,
        model: T.nn.Module,
        gpu_id: int,
        args: Dict,
        log_enabled: bool = True,
        is_eval: bool = False,
    ) -> None:
        self.model = model
        self.args = args
        self.gpu_id = gpu_id
        self.log_enabled = log_enabled
        self.is_eval = is_eval

        self.uid = int(time.time())

        if not is_eval:
            self.optim = T.optim.AdamW(
                self.model.parameters(),
                lr=args['lr'],
                betas=(0.9, 0.999),
            )
            self.scaler = T.cuda.amp.GradScaler()

            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optim,
                first_cycle_steps=args['cycle_steps'],
                cycle_mult=args['cycle_mult'],
                max_lr=args['lr'],
                min_lr=args['min_lr'],
                warmup_steps=args['warmup_steps'],
                gamma=args['lr_decay_factor'],
            )

        self.model = self.model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])

        self.logger = get_logger(__class__.__name__, gpu_id)
        if log_enabled and self.gpu_id == 0:
            self.args['log_dir'] = os.path.join(args['log_dir'], f'{self.uid}')
            self.summary_writer = SummaryWriter(log_dir=self.args['log_dir'])
            self.args['ckpt_dir'] = os.path.join(self.args['log_dir'], 'weights')
            os.makedirs(self.args['ckpt_dir'], exist_ok=True)
            self.save_config()

        self.last_loss = None
        self.last_metric_val = None
        self.counter = 0
        self.best_epoch = None
        self.best_metric_val = None

    def is_metric_val_better(self, epoch=None):
        if self.best_metric_val is None or self.last_metric_val > self.best_metric_val:
            self.best_metric_val = self.last_metric_val
            self.best_epoch = epoch
            return True
        return False

    def write_summary(self, title, value, step):
        if self.log_enabled and self.gpu_id == 0:
            self.summary_writer.add_scalar(title, value, step)
    
    def write_images(self, title, images, step):
        if self.log_enabled and self.gpu_id == 0:
            self.summary_writer.add_images(title, images, step)

    def step(
        self,
        img: T.Tensor,
        label: T.Tensor,
        mask: T.Tensor,
        valid: T.Tensor,
        seg_type: T.Tensor,
        is_train: bool,
    ):
        img = img.to(self.gpu_id)
        label = label.to(self.gpu_id)
        mask = mask.to(self.gpu_id)
        valid = valid.to(self.gpu_id)
        seg_type = seg_type.to(self.gpu_id)

        with T.cuda.amp.autocast():
            feature_ensemble = -1 # TODO: test change to 0 to enable
            loss, pred, bool_masked_pos = self.model(img, label, mask, valid, seg_type, feature_ensemble)

        if is_train:
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        
        return loss.item(), pred, bool_masked_pos
    
    def visualize(self, title:str, imgs: T.Tensor, labels: T.Tensor, preds: T.Tensor, discretized_cmaps: T.Tensor, masks: T.Tensor, counter:int = None):
        if self.log_enabled and self.gpu_id == 0 and (self.counter % self.args['image_log_interval'] == 0 or counter is not None):
            counter = counter if counter is not None else self.counter
            idx = np.random.randint(0, len(preds))

            input_img = imgs[idx].cpu()
            label = labels[idx].cpu()
            pred = preds[idx].cpu()
            cmap = discretized_cmaps[idx].cpu()
            mask = masks[idx].cpu().float()
            masked_label = (1 - mask) * label  + mask * T.randn_like(mask)
            
            result = T.concatenate([input_img, masked_label, label], axis=2)
            result = T.permute(result, (1, 2, 0))
            result = self.unnormalize(result)
            
            result = T.concatenate([result, pred, cmap / 255.0], axis=1)
            self.summary_writer.add_image(title, result, counter, dataformats='HWC')
    
    def unnormalize(self, img: T.tensor):
        # img B, H, W, 3 or H, W, 3
        std = T.FloatTensor(self.args['image_std']).to(img.device)
        mean = T.FloatTensor(self.args['image_mean']).to(img.device)
        return T.clip((img * std + mean), 0, 1)

    def iou(self, pred: T.tensor, label: T.tensor, mask: T.tensor, ori_label: T.tensor, color_palette: T.tensor):
        pred = self.model.module.unpatchify(pred) # B, 3, H, W
        mask = mask[:, :, None].repeat(1, 1, self.model.module.patch_size**2 * 3)
        mask = self.model.module.unpatchify(mask)
        mask = mask.to(pred.device)
        ori_label = ori_label.to(pred.device)
        color_palette = color_palette.to(pred.device)

        pred = T.permute(pred, (0, 2, 3, 1))  # B, H, W, 3
        pred = self.unnormalize(pred)
        discretized_cmap, pred_label = cmap_to_lbl(pred * 255.0, color_palette)
        n_class = color_palette.shape[1]
        result = calculate_iou(pred_label, ori_label, mask[:, 0, :, :], n_class)
        return result, pred, discretized_cmap, mask

    def process_data(self, dl: T.utils.data.DataLoader, is_train: bool, epoch: int):
        if is_train:
            self.logger.info('Training Phase')
        elif not self.is_eval:
            self.logger.info('Validation Phase')

        n_class = dl.dataset.max_classes
        iou = T.zeros((n_class, 2), device=self.gpu_id)
        batch_losses = T.zeros(2, device=self.gpu_id)

        pbar = tqdm(dl, disable=self.gpu_id != 0)

        for i, (img, label, mask, valid, seg_type, ori_label, color_palette) in enumerate(pbar):
            if not is_train:
                self.model.eval()
                with T.no_grad():
                    b_loss, b_pred, b_mask = self.step(img, label, mask, valid, seg_type, is_train)
                
                c_iou, preds, cmaps, masks = self.iou(b_pred, label, mask, ori_label, color_palette)
                val_counter = epoch * len(dl) + i
                if val_counter % self.args['image_log_interval'] == 0:
                    self.visualize(f'Validation/{epoch}', img, label, preds, cmaps, masks, val_counter)
            else:
                self.model.train()
                b_loss, b_pred, b_mask = self.step(img, label, mask, valid, seg_type, is_train)
                self.counter += 1

                self.scheduler.step()
                
                c_iou, preds, cmaps, masks = self.iou(b_pred, label, mask, ori_label, color_palette)

                self.write_summary(f'LR Scheduler', self.optim.param_groups[0]['lr'], self.counter)
                self.write_summary('Training/Batch Loss', b_loss, self.counter)

                self.visualize(f'Training/{epoch}', img, label, preds, cmaps, masks)
                yield i


            if self.gpu_id != 0:  # reset for gpu rank > 0
                batch_losses = T.zeros(2, device=self.gpu_id)
                iou = T.zeros((n_class, 2), device=self.gpu_id)

            batch_losses[0] += b_loss
            batch_losses[1] += 1
            iou = iou + c_iou
            
            T.distributed.reduce(batch_losses, dst=0)
            T.distributed.reduce(iou, dst=0)

            avg_losses = batch_losses[0] / batch_losses[1]
            m_iou = 100 * iou[:, 0] / (iou[:, 1] + 1e-10)
            
            # if is_train: # when training, only check iou for the first 7 classes (base)
            #     m_iou = m_iou[:7].mean().item()
            # else: # when validation, only check iou for the last 4 classes (novel)
            #     m_iou = m_iou[:-4].mean().item()

            pbar.set_postfix({
                'Loss': f'{avg_losses:.5f}',
                'mIoU': f'{m_iou[1:].mean().item():.3f}', # exclude background on mIoU
                'IoU': ['%.3f' % x for x in m_iou.tolist()]
            })

        if not is_train:
            self.last_loss = avg_losses
            self.last_metric_val = m_iou[1:].mean().item()

            self.write_summary('Validation/Loss', avg_losses, epoch)
            self.write_summary('Validation/mIoU', m_iou[1:].mean().item(), epoch)
        else:
            self.write_summary('Training/Loss', avg_losses, epoch)
            self.write_summary('Training/mIoU', m_iou[1:].mean().item(), epoch)

        yield -1

    def save_config(self):
        config = self.args

        self.logger.info('======CONFIGURATIONS======')
        for k, v in config.items():
            self.logger.info(f'{k.upper()}: {v}')

        config_path = os.path.join(self.args['log_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        self.logger.info(f'Training config saved to {config_path}')

    def save_checkpoint(self, epoch: int, name: str = '', only_model: bool = True):
        if self.gpu_id == 0:
            save_checkpoint = {'model_state_dict': self.model.module.state_dict()}
            if not only_model:
                save_checkpoint['optimizer_state_dict'] = self.optim.state_dict()
                save_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if name != '':
                ckpt_path = os.path.join(self.args['ckpt_dir'], f'{name}.pt')
            else:
                ckpt_path = os.path.join(
                    self.args['ckpt_dir'],
                    f'epoch{epoch:02}_loss{self.last_loss:.4f}_metric{self.last_metric_val:.4f}.pt',
                )
            T.save(save_checkpoint, ckpt_path)
            self.logger.info(f'Checkpoint saved to {ckpt_path}')

    def load_checkpoint(self, ckpt_path: str, only_model: bool = True):
        assert os.path.exists(ckpt_path)
        checkpoint = T.load(ckpt_path, map_location='cpu')
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        if not only_model:
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logger.info(f'Succesfully loaded model in {ckpt_path}')

    def do_training(
        self,
        train_dataloader: T.utils.data.DataLoader,
        val_dataloader: T.utils.data.DataLoader,
        eval_per_epoch: int = 1,
    ):
        eval_idx = [len(train_dataloader) // eval_per_epoch * i for i in range(1, eval_per_epoch)]
        epoch = self.args['epoch']
        for i in range(epoch):
            self.logger.info(f'Epoch {i+1}/{epoch}')
            k = 0
            for step in self.process_data(train_dataloader, True, i):
                if step in eval_idx or step == -1:
                    deque(self.process_data(val_dataloader, False, eval_per_epoch * i + k), maxlen=0)

                    if self.is_metric_val_better(i + 1):
                        self.save_checkpoint(i + 1, 'best')
                    k += 1

            if (i + 1) % self.args['ckpt_interval'] == 0 or i == self.args['epoch'] - 1:
                self.save_checkpoint(i + 1)

            self.logger.info(f'Epoch complete\n')
        self.logger.info(f'Best result was seen in epoch {self.best_epoch}')

    def do_evaluation(self, test_dataloader: T.utils.data.DataLoader):
        deque(self.process_data(test_dataloader, False, 0), maxlen=0)
        self.logger.info(f'Loss: {self.last_loss:.5f}')

class AgentAdapter(Agent):
    def step(
        self,
        img: T.Tensor,
        label: T.Tensor,
        mask: T.Tensor,
        valid: T.Tensor,
        seg_type: T.Tensor,
        is_train: bool,
    ):
        img = img.to(self.gpu_id)
        label = label.to(self.gpu_id)
        mask = mask.to(self.gpu_id)
        valid = valid.to(self.gpu_id)
        seg_type = seg_type.to(self.gpu_id)

        with T.cuda.amp.autocast():
            feature_ensemble = -1 # TODO: test change to 0 to enable
            loss, pred, bool_masked_pos = self.model(img, label, mask, valid, seg_type, feature_ensemble)

        if is_train:
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        
        return loss.item(), pred, bool_masked_pos
    
    def visualize(self, title:str, imgs: T.Tensor, labels: T.Tensor, preds: T.Tensor, discretized_cmaps: T.Tensor, masks: T.Tensor, counter:int = None):
        if self.log_enabled and self.gpu_id == 0 and (self.counter % self.args['image_log_interval'] == 0 or counter is not None):
            counter = counter if counter is not None else self.counter
            idx = np.random.randint(0, len(preds))

            input_img = imgs[idx].cpu()
            label = labels[idx].cpu()
            pred = preds[idx].cpu()
            cmap = discretized_cmaps[idx].cpu()
            mask = masks[idx].cpu().float()
            masked_label = (1 - mask) * label  + mask * T.randn_like(mask)
            
            result = T.concatenate([input_img, masked_label, label], axis=2)
            result = T.permute(result, (1, 2, 0))
            result = self.unnormalize(result)
            
            result = T.concatenate([result, pred, cmap / 255.0], axis=1)
            self.summary_writer.add_image(title, result, counter, dataformats='HWC')
    
    def save_checkpoint(self, epoch: int, name: str = '', only_model: bool = True):
        if self.gpu_id == 0:
            save_checkpoint = {'model_state_dict': self.model.module.state_dict()}
            if not only_model:
                save_checkpoint['optimizer_state_dict'] = self.optim.state_dict()
                save_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if name != '':
                ckpt_path = os.path.join(self.args['ckpt_dir'], f'{name}.pt')
            else:
                ckpt_path = os.path.join(
                    self.args['ckpt_dir'],
                    f'epoch{epoch:02}_loss{self.last_loss:.4f}_metric{self.last_metric_val:.4f}.pt',
                )
            T.save(save_checkpoint, ckpt_path)
            self.logger.info(f'Checkpoint saved to {ckpt_path}')

    def process_data(self, dl: T.utils.data.DataLoader, is_train: bool, epoch: int):
        if is_train:
            self.logger.info('Training Phase')
        elif not self.is_eval:
            self.logger.info('Validation Phase')

        n_class = dl.dataset.max_classes
        iou = T.zeros((n_class, 2), device=self.gpu_id)
        batch_losses = T.zeros(2, device=self.gpu_id)

        pbar = tqdm(dl, disable=self.gpu_id != 0)

        for i, (img, label, mask, valid, seg_type, ori_label, color_palette) in enumerate(pbar):
            if not is_train:
                self.model.eval()
                with T.no_grad():
                    b_loss, b_pred, b_mask = self.step(img, label, mask, valid, seg_type, is_train)
                
                c_iou, preds, cmaps, masks = self.iou(b_pred, label, mask, ori_label, color_palette)
                val_counter = epoch * len(dl) + i
                if val_counter % self.args['image_log_interval'] == 0:
                    self.visualize(f'Validation/{epoch}', img, label, preds, cmaps, masks, val_counter)
            else:
                self.model.train()
                b_loss, b_pred, b_mask = self.step(img, label, mask, valid, seg_type, is_train)
                self.counter += 1

                self.scheduler.step()
                
                c_iou, preds, cmaps, masks = self.iou(b_pred, label, mask, ori_label, color_palette)

                self.write_summary(f'LR Scheduler', self.optim.param_groups[0]['lr'], self.counter)
                self.write_summary('Training/Batch Loss', b_loss, self.counter)

                self.visualize(f'Training/{epoch}', img, label, preds, cmaps, masks)
                yield i


            if self.gpu_id != 0:  # reset for gpu rank > 0
                batch_losses = T.zeros(2, device=self.gpu_id)
                iou = T.zeros((n_class, 2), device=self.gpu_id)

            batch_losses[0] += b_loss
            batch_losses[1] += 1
            iou = iou + c_iou
            
            T.distributed.reduce(batch_losses, dst=0)
            T.distributed.reduce(iou, dst=0)

            avg_losses = batch_losses[0] / batch_losses[1]
            m_iou = 100 * iou[:, 0] / (iou[:, 1] + 1e-10)
            
            # if is_train: # when training, only check iou for the first 7 classes (base)
            #     m_iou = m_iou[:7].mean().item()
            # else: # when validation, only check iou for the last 4 classes (novel)
            #     m_iou = m_iou[:-4].mean().item()

            pbar.set_postfix({
                'Loss': f'{avg_losses:.5f}',
                'mIoU': f'{m_iou[1:].mean().item():.3f}', # exclude background on mIoU
                'IoU': ['%.3f' % x for x in m_iou.tolist()]
            })

        if not is_train:
            self.last_loss = avg_losses
            self.last_metric_val = m_iou[1:].mean().item()

            self.write_summary('Validation/Loss', avg_losses, epoch)
            self.write_summary('Validation/mIoU', m_iou[1:].mean().item(), epoch)
        else:
            self.write_summary('Training/Loss', avg_losses, epoch)
            self.write_summary('Training/mIoU', m_iou[1:].mean().item(), epoch)

        yield -1



class AgentOneClass(Agent):
    def visualize(self, title:str, imgs: T.Tensor, labels: T.Tensor, preds: T.Tensor, discretized_cmaps: T.Tensor, masks: T.Tensor, class_label:T.tensor, counter:int = None):
        if self.log_enabled and self.gpu_id == 0 and (self.counter % self.args['image_log_interval'] == 0 or counter is not None):
            counter = counter if counter is not None else self.counter
            idx = np.random.randint(0, len(preds))

            input_img = imgs[idx].cpu()
            label = labels[idx].cpu()
            pred = preds[idx].cpu()
            cmap = discretized_cmaps[idx].cpu()
            mask = masks[idx].cpu().float()
            masked_label = (1 - mask) * label  + mask * T.randn_like(mask)
            c_label = class_label.cpu()[idx]
            
            result = T.concatenate([input_img, masked_label, label], axis=2)
            result = T.permute(result, (1, 2, 0))
            result = self.unnormalize(result)
            
            result = T.concatenate([result, pred, cmap / 255.0], axis=1)
            self.summary_writer.add_image(f'{title}/{c_label}', result, counter, dataformats='HWC')

    def iou(self, pred: T.tensor, label: T.tensor, mask: T.tensor, ori_label: T.tensor, color_palette: T.tensor, n_classes: int, class_label: T.tensor):
        pred = self.model.module.unpatchify(pred) # B, 3, H, W
        mask = mask[:, :, None].repeat(1, 1, self.model.module.patch_size**2 * 3)
        mask = self.model.module.unpatchify(mask)
        mask = mask.to(pred.device)
        ori_label = ori_label.to(pred.device)
        color_palette = color_palette.to(pred.device)
        class_label = class_label.to(pred.device)

        pred = T.permute(pred, (0, 2, 3, 1))  # B, H, W, 3
        pred = self.unnormalize(pred)
        discretized_cmap, pred_label = cmap_to_lbl(pred * 255.0, color_palette)
        result = calculate_iou_one_class(pred_label, ori_label, mask[:, 0, :, :], n_classes, class_label)
        return result, pred, discretized_cmap, mask

    def step(
        self,
        img: T.Tensor,
        label: T.Tensor,
        mask: T.Tensor,
        valid: T.Tensor,
        seg_type: T.Tensor,
        class_weight: T.Tensor,
        is_train: bool,
    ):
        img = img.to(self.gpu_id)
        label = label.to(self.gpu_id)
        mask = mask.to(self.gpu_id)
        valid = valid.to(self.gpu_id)
        seg_type = seg_type.to(self.gpu_id)
        class_weight = class_weight.to(self.gpu_id)

        with T.cuda.amp.autocast():
            feature_ensemble = -1 # TODO: test change to 0 to enable
            loss, pred, bool_masked_pos = self.model(img, label, mask, valid, seg_type, feature_ensemble)
        final_loss = (loss * class_weight).mean()

        if is_train:
            self.optim.zero_grad()
            self.scaler.scale(final_loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        
        return final_loss.item(), pred, bool_masked_pos

    def process_data(self, dl: T.utils.data.DataLoader, is_train: bool, epoch: int):
        if is_train:
            self.logger.info('Training Phase')
        elif not self.is_eval:
            self.logger.info('Validation Phase')

        n_classes = dl.dataset.max_classes
        iou = T.zeros((n_classes, 2), device=self.gpu_id)
        batch_losses = T.zeros(2, device=self.gpu_id)

        pbar = tqdm(dl, disable=self.gpu_id != 0)

        for i, (img, label, mask, valid, seg_type, ori_label, color_palette, class_weight, class_label) in enumerate(pbar):
            if not is_train:
                self.model.eval()
                with T.no_grad():
                    b_loss, b_pred, b_mask = self.step(img, label, mask, valid, seg_type, class_weight, is_train)
                
                c_iou, preds, cmaps, masks = self.iou(b_pred, label, mask, ori_label, color_palette, n_classes, class_label)
                val_counter = epoch * len(dl) + i
                self.visualize(f'Validation', img, label, preds, cmaps, masks, counter=val_counter, class_label=class_label)
            else:
                self.model.train()
                b_loss, b_pred, b_mask = self.step(img, label, mask, valid, seg_type, class_weight, is_train)
                self.counter += 1

                self.scheduler.step()
                
                c_iou, preds, cmaps, masks = self.iou(b_pred, label, mask, ori_label, color_palette, n_classes, class_label)

                self.write_summary(f'LR Scheduler', self.optim.param_groups[0]['lr'], self.counter)
                self.write_summary('Training/Batch Loss', b_loss, self.counter)

                self.visualize(f'Training', img, label, preds, cmaps, masks, class_label=class_label)
                yield i


            if self.gpu_id != 0:  # reset for gpu rank > 0
                batch_losses = T.zeros(2, device=self.gpu_id)
                iou = T.zeros((n_classes, 2), device=self.gpu_id)

            batch_losses[0] += b_loss
            batch_losses[1] += 1
            iou = iou + c_iou
            
            T.distributed.reduce(batch_losses, dst=0)
            T.distributed.reduce(iou, dst=0)

            avg_losses = batch_losses[0] / batch_losses[1]
            m_iou = 100 * iou[:, 0] / (iou[:, 1] + 1e-10)
            base_m_iou = m_iou[1:8].mean()
            novel_m_iou = m_iou[8:].mean()
            weighted_m_iou = 0.4 * base_m_iou + 0.6 * novel_m_iou
            
            pbar.set_postfix({
                'Loss': f'{avg_losses:.5f}',
                'B mIoU': f'{base_m_iou:.3f}',
                'N mIoU': f'{novel_m_iou:.3f}',
                'W mIoU': f'{weighted_m_iou:.3f}',
                'IoU': ['%.3f' % x for x in m_iou.tolist()]
            })

        if not is_train:
            self.last_loss = avg_losses
            self.last_metric_val = weighted_m_iou

            self.write_summary('Validation/Loss', avg_losses, epoch)
            self.write_summary('Validation/mIoU', weighted_m_iou, epoch)
        else:
            self.write_summary('Training/Loss', avg_losses, epoch)
            self.write_summary('Training/mIoU', weighted_m_iou, epoch)

        yield -1