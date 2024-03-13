import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import os
import json
import torch as T
import torch.multiprocessing as mp
import argparse
import torch.nn.functional as F
from agent import AgentOneClass
from typing import Dict
from utils import *
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from Painter.SegGPT.SegGPT_inference.models_seggpt import SegGPT
from data import OEMOneClassDataset, OEMOneClassSMDataset
from functools import partial

class SegGPTWithLoss(SegGPT):
    def forward_loss(self, pred, tgts, mask, valid):
        """
        tgts: [N, 3, H, W]
        pred: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        valid: [N, 3, H, W]
        """
        mask = mask[:, :, None].repeat(1, 1, self.patch_size**2 * 3)
        mask = self.unpatchify(mask)
        mask = mask * valid

        target = tgts
        if self.loss_func == "l1l2":
            loss = ((pred - target).abs() + (pred - target) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - target).abs()
        elif self.loss_func == "l2":
            loss = (pred - target) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        loss = (loss * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))  # mean loss for each batch
        return loss

def get_model(**kwargs):
    model = SegGPTWithLoss(
        img_size=(896, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(T.nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        **kwargs)
    return model

def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    T.cuda.set_device(rank)
    T.cuda.empty_cache()
    init_process_group('nccl', rank=rank, world_size=world_size)


def main(rank: int, world_size: int, train_args: Dict):
    ddp_setup(rank, world_size)

    setup_logging()
    logger = get_logger(__name__, rank)

    logger.info('Preparing dataset')
    train_dataset = OEMOneClassSMDataset(
        root = train_args['train_dataset_dir'], 
        max_classes = train_args['n_classes'],
        mean = train_args['image_mean'],
        std = train_args['image_std'],
        mask_ratio = train_args['mask_ratio'],
        negative_pairs_ratio = train_args['negative_pairs_ratio'],
        validation_ratio = train_args['validation_ratio'],
        include_class = train_args['include_class'],
        is_train=True,
    )
    val_dataset = OEMOneClassSMDataset(
        root = train_args['val_dataset_dir'], 
        max_classes = train_args['n_classes'],
        mean = train_args['image_mean'],
        std = train_args['image_std'],
        mask_ratio = train_args['mask_ratio'],
        negative_pairs_ratio = train_args['negative_pairs_ratio'],
        validation_ratio = train_args['validation_ratio'],
        include_class = train_args['include_class'],
        is_train = False,
    )

    logger.info('Instantiating model and trainer agent')

    model = get_model()
    initial_ckpt = T.load('seggpt_vit_large.pth', map_location='cpu')
    model.load_state_dict(initial_ckpt['model'], strict=False)
    logger.info('Initial checkpoint loaded')

    trainer = AgentOneClass(model, rank, train_args)

    logger.info(f'Using {T.cuda.device_count()} GPU(s)')
    if 'model_path' in train_args:
        trainer.load_checkpoint(train_args['model_path'])

    logger.info('Instantiating dataloader')
    train_dataloader = T.utils.data.DataLoader(
        train_dataset,
        batch_size=train_args['batch_size'],
        shuffle=False,
        num_workers=train_args['num_workers'],
        pin_memory=True,
        sampler=DistributedSampler(train_dataset),
    )
    val_dataloader = T.utils.data.DataLoader(
        val_dataset,
        batch_size=train_args['batch_size'],
        shuffle=False,
        num_workers=train_args['num_workers'],
        pin_memory=True,
        sampler=DistributedSampler(val_dataset),
    )

    trainer.do_training(train_dataloader, val_dataloader, train_args['eval_per_epoch'])
    destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to json config', default='base.json')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    train_args = json.load(open(args.config, 'r'))
    world_size = T.cuda.device_count()
    mp.spawn(main, nprocs=world_size, args=(world_size, train_args))
