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
def run_one_image(img, tgt, model, device, mask=None):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    if mask is None:
        bool_masked_pos = torch.zeros(model.seggpt.patch_embed.num_patches)
        bool_masked_pos[model.seggpt.patch_embed.num_patches//2:] = 1
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    else:
        bool_masked_pos = torch.tensor(mask).unsqueeze(dim=0)
    valid = torch.ones(tgt.shape[0], tgt.shape[1], tgt.shape[2] * 2, tgt.shape[3])
    seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
    y = model.seggpt.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    
    mask = mask[:, :, None].repeat(1, 1, model.seggpt.patch_size**2 * 3)
    mask = model.seggpt.unpatchify(mask)
    # print('m', mask.shape)
    mask = mask.permute(0, 2, 3, 1)
    mask = mask[0, mask.shape[1]//2:, :, :]
    mask = mask.cpu().float()
    
    # temp_img = x[0].permute(1, 2, 0)
    # # print(temp_img.shape)
    # # masked_label = (1 - mask) * temp_img  + mask * torch.randn_like(mask)
    # # masked_label = torch.clip((masked_label * imagenet_std + imagenet_mean) * 255, 0, 255)
    # masked_label = torch.clip((temp_img * imagenet_std + imagenet_mean) * 255, 0, 255)
    # masked_label = masked_label.cpu().numpy()
    # print(masked_label.shape)
    # timg = Image.fromarray((masked_label).astype(np.uint8))
    # timg.save('temp_test.png')


    return output, mask

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
            output, _ = run_one_image(image, tgt, model, device)
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


def create_stitch_mask(h, w, type, width):
    prompt_mask = np.zeros(h * w)
    image_mask = np.zeros((h, w))
    if type == 0:
        image_mask[:, image_mask.shape[1] // 2 - width: image_mask.shape[1] // 2 + width] = 1
    elif type == 1:
        image_mask[image_mask.shape[0] // 2 - width: image_mask.shape[0] // 2 + width, :] = 1
    else:
        image_mask[image_mask.shape[0] // 2 - width: image_mask.shape[0] // 2 + width, image_mask.shape[1] // 2 - width: image_mask.shape[1] // 2 + width] = 1
    image_mask = image_mask.flatten()
    result = np.concatenate((prompt_mask, image_mask))
    return result

def inference_stitch(model, device, img_path, tgt_path, lbl_path, out_dir, split=2, width=4):
    # run after inference_image_with_crop
    # only works for split = 2
    res, hres = 448, 448

    full_image = Image.open(img_path).convert('RGB').resize((1024, 1024))
    full_tgt = Image.open(tgt_path).convert('RGB').resize((1024, 1024), Image.NEAREST)
    full_lbl = Image.open(lbl_path).convert('L').resize((1024, 1024), Image.NEAREST)
    col_size = full_image.size[0] // split
    row_size = full_image.size[1] // split
    
    w, h = full_image.size
    final_out_color = np.array(full_tgt)
    final_out_label = np.array(full_lbl)

    crop_params = [
        [(w // 4, 0, 3 * w // 4, h // 2), 0], # top middle
        [(w // 4, h // 2, 3 * w // 4, h), 0], # bottom middle
        [(0, h // 4, w // 2, 3 * h // 4), 1], # left middle
        [(w // 2, h // 4, w, 3 * h // 4), 1], # right middle
        [(w // 4, h // 4, 3 * w // 4, 3 * h // 4), 2] # center
    ]

    for i, (crop_param, stitch_type) in enumerate(crop_params):
        j1, i1, j2, i2 = crop_param
        assert j2 - j1 == col_size and i2 - i1 == row_size

        cropped_image = full_image.crop(crop_param).resize((res, hres))
        cropped_tgt = full_tgt.crop(crop_param).resize((res, hres), Image.NEAREST)

        # cropped_image.save(f'temp_cropped_image{i}.png')
        # cropped_tgt.save(f'temp_cropped_tgt{i}.png')

        img = np.array(cropped_image) / 255.
        img = (img - imagenet_mean) / imagenet_std
        img = np.expand_dims(img, axis=0)
        tgt = np.array(cropped_tgt) / 255.
        tgt = (tgt - imagenet_mean) / imagenet_std
        tgt = np.expand_dims(tgt, axis=0)

        torch.manual_seed(2)
        hstitch_mask = create_stitch_mask(28, 28, stitch_type, width)
        output, mask = run_one_image(img, tgt, model, device, hstitch_mask)
        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2), 
            size=[row_size, col_size], 
            mode='nearest',
        ).permute(0, 2, 3, 1)
        mask = F.interpolate(
            mask[None, ...].permute(0, 3, 1, 2), 
            size=[row_size, col_size], 
            mode='nearest',
        ).permute(0, 2, 3, 1)
        output, label = cmap_to_lbl(output, torch.tensor(color_map, device=output.device, dtype=output.dtype).unsqueeze(0))
        output = output[0].numpy()
        label = label[0].numpy()
        mask = mask[0].numpy()

        final_out_color[i1:i2, j1:j2] = output * mask + final_out_color[i1:i2, j1:j2] * (1 - mask)
        final_out_label[i1:i2, j1:j2] = label * mask[:, :, 0] * class_idx + final_out_label[i1:i2, j1:j2] * (1 - mask[:, :, 0])

    
    filename = os.path.basename(img_path).replace('.tif', '.png')
    os.makedirs(os.path.join(out_dir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'concat'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'label'), exist_ok=True)

    final_out_color = Image.fromarray((final_out_color).astype(np.uint8))
    final_out_color.save(os.path.join(out_dir, 'color', filename))

    final_out_label = Image.fromarray((final_out_label).astype(np.uint8))
    final_out_label.save(os.path.join(out_dir, 'label', filename))

    concat = np.concatenate((np.array(full_image), np.array(full_tgt), final_out_color), axis=1)
    concat = Image.fromarray((concat).astype(np.uint8))
    concat.save(os.path.join(out_dir, 'concat', filename))

model_path = 'logs/1710835935/weights/best.pt'
seggpt_model = seggpt_vit_large_patch16_input896x448()
model = AdapterSegGPT(seggpt_model)
print('Frozen model loaded')

ckpt = T.load(model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
print('Checkpoint loaded')

model = model.to(0)
model.eval()

dataset_dir = '/disk3/steve/dataset/OpenEarthMap-FSS/testset/images'
outdir = 'class_novel/11_phase2'
class_idx = 11
split = 2
for file in tqdm(os.listdir(dataset_dir)):
    inference_image_with_crop(model, 'cuda', os.path.join(dataset_dir, file), class_idx=class_idx, outdir=outdir, split=split)
    tgt_path = os.path.join(outdir, 'color', file.replace('.tif', '.png'))
    lbl_path = os.path.join(outdir, 'label', file.replace('.tif', '.png'))
    outdir_stitch = f'{outdir}_stitch'
    inference_stitch(model, 'cuda', os.path.join(dataset_dir, file), tgt_path, lbl_path, outdir_stitch, split=split, width=4)