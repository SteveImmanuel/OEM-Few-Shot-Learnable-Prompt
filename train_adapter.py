import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import argparse
import os
import json
import torch as T
import torch.multiprocessing as mp
from agent import AgentAdapter
from typing import Dict
from utils import *
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from Painter.SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448
from model import AdapterSegGPT
from data import OEMAdapterDataset, OEMAdapterDatasetV2

def ddp_setup(rank: int, world_size: int, port:int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    T.cuda.set_device(rank)
    T.cuda.empty_cache()
    init_process_group('nccl', rank=rank, world_size=world_size)


def main(rank: int, world_size: int, train_args: Dict, port: int):
    ddp_setup(rank, world_size, port)

    setup_logging()
    logger = get_logger(__name__, rank)

    logger.info('Preparing dataset')
    # train_dataset = OEMAdapterDataset(
    #     root = train_args['train_dataset_dir'],
    #     mean = train_args['image_mean'],
    #     std = train_args['image_std'],
    #     resize = (1024, 1024),
    #     is_train=True,
    # )
    # val_dataset = OEMAdapterDataset(
    #     root = train_args['val_dataset_dir'], 
    #     mean = train_args['image_mean'],
    #     std = train_args['image_std'],
    #     resize = (448, 448),
    #     is_train = False,
    # )

    train_dataset = OEMAdapterDatasetV2(
        root = train_args['train_dataset_dir'],
        mean = train_args['image_mean'],
        std = train_args['image_std'],
        resize = (1024, 1024),
        smallest_crop_size=train_args['smallest_crop_size'], 
        biggest_crop_size=train_args['biggest_crop_size'],
        smallest_stride=train_args['smallest_stride'],
        is_train=True,
        is_phase_2=train_args['phase_2'],
    )
    val_dataset = OEMAdapterDatasetV2(
        root = train_args['val_dataset_dir'], 
        mean = train_args['image_mean'],
        std = train_args['image_std'],
        resize = (1024, 1024),
        smallest_crop_size=train_args['smallest_crop_size'], 
        biggest_crop_size=train_args['biggest_crop_size'],
        smallest_stride=train_args['smallest_stride'],
        is_train = False,
        is_phase_2=train_args['phase_2'],
    )

    logger.info('Instantiating model and trainer agent')

    seggpt_model = seggpt_vit_large_patch16_input896x448()
    initial_ckpt = T.load(train_args['model_path'], map_location='cpu')
    seggpt_model.load_state_dict(initial_ckpt['model_state_dict'], strict=False)
    model = AdapterSegGPT(seggpt_model)
    logger.info('Frozen model loaded')

    trainer = AgentAdapter(model, rank, train_args)
    if train_args.get('adapter_path') is not None:
        trainer.load_checkpoint(train_args['adapter_path'])
    logger.info(f'Using {T.cuda.device_count()} GPU(s)')

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
    parser = argparse.ArgumentParser('SegGPT train adapter', add_help=False)
    parser.add_argument('--config', type=str, help='path to json config', required=True)
    parser.add_argument('--port', type=int, help='DDP port', default=12355)
    parser.add_argument('--phase-2', action='store_true', help='phase 2 training, positive negative samples')
    parser.add_argument('--adapter-path', type=str, help='path to adapter checkpoint')
    parser.add_argument('--uid', type=str, help='unique id for the run', default=None)
    parser.add_argument('--lr', type=float, help='learning rate', default=None)
    parser.add_argument('--epoch', type=int, help='epoch', default=None)
    parser.add_argument('--ckpt-interval', type=int, help='checkpoint interval (in epoch)', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    train_args = json.load(open(args.config, 'r'))
    train_args['adapter_path'] = args.adapter_path
    train_args['phase_2'] = args.phase_2
    if args.uid is not None:
        train_args['uid'] = args.uid
    if args.lr is not None:
        train_args['lr'] = args.lr
    if args.epoch is not None:
        train_args['epoch'] = args.epoch
    if args.ckpt_interval is not None:
        train_args['ckpt_interval'] = args.ckpt_interval
    world_size = T.cuda.device_count()
    mp.spawn(main, nprocs=world_size, args=(world_size, train_args, args.port))
