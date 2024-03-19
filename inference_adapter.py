import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import os
import json
import torch as T
import numpy as np
from agent import AgentAdapter
from typing import Dict
from utils import *
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from Painter.SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448
from model import AdapterSegGPT
from data import OEMAdapterDataset
from PIL import Image
from tqdm import tqdm


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

color_map = np.array([
    (0, 0,  0),
    (255, 255, 255),
])

def unnormalize(img: T.tensor, image_std, image_mean):
    # img B, H, W, 3 or H, W, 3
    std = T.FloatTensor(image_std).to(img.device)
    mean = T.FloatTensor(image_mean).to(img.device)
    return T.clip((img * std + mean), 0, 1)

def to_img_tensor(arr: np.ndarray, image_std, image_mean):
    arr = arr / 255.0
    arr = arr - image_mean
    arr = arr / image_std
    res = torch.FloatTensor(arr)
    res = torch.einsum('hwc->chw', res)
    return res



@torch.no_grad()
def run_one_image(img, tgt, model, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.seggpt.patch_embed.num_patches)
    bool_masked_pos[model.seggpt.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones(tgt.shape[0], tgt.shape[1], tgt.shape[2] * 2, tgt.shape[3])
    seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
    y = model.seggpt.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output

def inference_image_with_crop(model, device, img_path, class_idx, outdir, split=2):
    res, hres = 448, 448

    full_image = Image.open(img_path).convert('RGB').resize((1024, 1024))
    row_size = full_image.size[0] // split
    col_size = full_image.size[1] // split
    
    h, w = full_image.size
    final_out_color = np.zeros((h, w, 3))
    final_out_label = np.zeros((h, w))
    final_out_image = np.zeros((h, w, 3))

    for row in range(split):
        for col in range(split):
            image = full_image.crop((row * row_size, col * col_size, (row + 1) * row_size, (col + 1) * col_size))
            input_image = np.array(image)

            image = np.array(image.resize((res, hres))) / 255.
            image = (image - imagenet_mean) / imagenet_std
            image = np.expand_dims(image, axis=0)
            tgt = np.zeros_like(image)

            torch.manual_seed(2)
            output = run_one_image(image, tgt, model, device)
            output = F.interpolate(
                output[None, ...].permute(0, 3, 1, 2), 
                size=[row_size, col_size], 
                mode='nearest',
            ).permute(0, 2, 3, 1)
            output, label = cmap_to_lbl(output, torch.tensor(color_map, device=output.device, dtype=output.dtype).unsqueeze(0))
            output = output[0].numpy()
            label = label[0].numpy()
            final_out_color[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = output
            final_out_label[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = label
            final_out_image[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = input_image


    concat = np.concatenate((final_out_image, final_out_color), axis=1)
    final_out_label = final_out_label * class_idx
    final_out_color = Image.fromarray((final_out_color).astype(np.uint8))
    concat = Image.fromarray((concat).astype(np.uint8))
    final_out_label = Image.fromarray((final_out_label).astype(np.uint8))

    os.makedirs(os.path.join(outdir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'concat'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'label'), exist_ok=True)
    filename = os.path.basename(img_path)

    final_out_color.save(os.path.join(outdir, 'color', filename.replace('.tif', '.png')))
    concat.save(os.path.join(outdir, 'concat', filename.replace('.tif', '.png')))
    final_out_label.save(os.path.join(outdir, 'label', filename.replace('.tif', '.png')))

setup_logging()
logger = get_logger(__name__, 0)
model_path = 'logs/1710831235/weights/best.pt'
seggpt_model = seggpt_vit_large_patch16_input896x448()
initial_ckpt = T.load(model_path, map_location='cpu')
seggpt_model.load_state_dict(initial_ckpt['model_state_dict'], strict=False)
model = AdapterSegGPT(seggpt_model)
logger.info('Frozen model loaded')

ckpt = T.load(model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
logger.info('Checkpoint loaded')

model = model.to(0)
model.eval()

dataset_dir = '/disk3/steve/dataset/OpenEarthMap-FSS/testset/images'
outdir = 'class_novel/8_phase2'
split = 4
for file in tqdm(os.listdir(dataset_dir)):
    inference_image_with_crop(model, 'cuda', os.path.join(dataset_dir, file), class_idx=9, outdir=outdir, split=split)
    # full_image = Image.open(os.path.join(dataset_dir, file)).resize((896, 896))
    # row_size = full_image.size[0] // split
    # col_size = full_image.size[1] // split

    # h, w = full_image.size
    # final_out_color = np.zeros((h, w, 3))
    # # final_out_label = np.zeros((h, w))
    # final_out_image = np.zeros((h, w, 3))

    # for row in range(split):
    #     for col in range(split):
    #         c_img = full_image.crop((row * row_size, col * col_size, (row + 1) * row_size, (col + 1) * col_size))
    #         c_img = np.array(c_img.resize((448, 448))).astype(np.uint8)
    #         img = to_img_tensor(c_img, train_args['image_std'], train_args['image_mean'])

    #         img = img.unsqueeze(0).to(0)

    #         zero_label = T.zeros_like(img)
    #         with T.no_grad():
    #             _, pred, _ = model(img, zero_label, mask, valid, seg_type, -1)
            
    #         pred = model.seggpt.unpatchify(pred)
    #         print(pred.shape)
    #         img = img[0]
    #         pred = pred[0]
    #         pred = pred[:, pred.shape[1]//2:, :]
    #         print(img.shape, pred.shape)
    #         print(row, col)

    #         pred = pred.permute(1, 2, 0)
    #         pred = pred.cpu().numpy()
            
    #         final_out_color[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = pred
    #         # final_out_label[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = label
    #         final_out_image[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = c_img

    # res = np.concatenate((final_out_image, final_out_color), axis=1)
    # res = (res * train_args['image_std'] + train_args['image_mean']) * 255
    # # res = unnormalize(res, train_args['image_std'], train_args['image_mean'])
    # # res_numpy = res.cpu().numpy()
    # img = Image.fromarray(res.astype(np.uint8))
    # img.save(f'restemp/res_{file}.png')

