import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import os
import torch as T
import argparse
import numpy as np
from typing import Dict
from utils import *
import torch.nn.functional as F
from Painter.SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448
from model import AdapterSegGPT
from PIL import Image
from tqdm import tqdm


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMANGENET_STD = np.array([0.229, 0.224, 0.225])

COLOR_MAP = np.array([
    (0, 0,  0),
    (255, 255, 255),
])

@torch.no_grad()
def run_one_image(img, tgt, model, device, mask=None):
    x = torch.tensor(img)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
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
    output = torch.clip((output * IMANGENET_STD + IMAGENET_MEAN) * 255, 0, 255)
    
    mask = mask[:, :, None].repeat(1, 1, model.seggpt.patch_size**2 * 3)
    mask = model.seggpt.unpatchify(mask)
    mask = mask.permute(0, 2, 3, 1)
    mask = mask[0, mask.shape[1]//2:, :, :]
    mask = mask.cpu().float()

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
            image = (image - IMAGENET_MEAN) / IMANGENET_STD
            image = np.expand_dims(image, axis=0)
            tgt = np.zeros_like(image)

            torch.manual_seed(2)
            output, _ = run_one_image(image, tgt, model, device)
            output = F.interpolate(
                output[None, ...].permute(0, 3, 1, 2), 
                size=[row_size, col_size], 
                mode='nearest',
            ).permute(0, 2, 3, 1)
            output, label = cmap_to_lbl(output, torch.tensor(COLOR_MAP, device=output.device, dtype=output.dtype).unsqueeze(0))
            output = output[0].numpy()
            label = label[0].numpy()
            final_out_color[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = output
            final_out_label[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = label
            final_out_image[col * col_size:(col + 1) * col_size, row * row_size:(row + 1) * row_size] = input_image

    final_out_image = np.clip(final_out_image * 0.6 + final_out_color * 0.4, 0, 255)
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

def inference_stitch(model, device, img_path, class_idx, tgt_path, lbl_path, outdir, split=2, width=4):
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

        img = np.array(cropped_image) / 255.
        img = (img - IMAGENET_MEAN) / IMANGENET_STD
        img = np.expand_dims(img, axis=0)
        tgt = np.array(cropped_tgt) / 255.
        tgt = (tgt - IMAGENET_MEAN) / IMANGENET_STD
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
        output, label = cmap_to_lbl(output, torch.tensor(COLOR_MAP, device=output.device, dtype=output.dtype).unsqueeze(0))
        output = output[0].numpy()
        label = label[0].numpy()
        mask = mask[0].numpy()

        final_out_color[i1:i2, j1:j2] = output * mask + final_out_color[i1:i2, j1:j2] * (1 - mask)
        final_out_label[i1:i2, j1:j2] = label * mask[:, :, 0] * class_idx + final_out_label[i1:i2, j1:j2] * (1 - mask[:, :, 0])

    
    filename = os.path.basename(img_path).replace('.tif', '.png')
    os.makedirs(os.path.join(outdir, 'stitch', 'color'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'stitch', 'concat'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'stitch', 'label'), exist_ok=True)

    final_out_color = Image.fromarray((final_out_color).astype(np.uint8))
    final_out_color.save(os.path.join(outdir, 'stitch', 'color', filename))

    final_out_label = Image.fromarray((final_out_label).astype(np.uint8))
    final_out_label.save(os.path.join(outdir, 'stitch', 'label', filename))

    concat = np.concatenate((np.array(full_image), np.array(full_tgt), final_out_color), axis=1)
    concat = Image.fromarray((concat).astype(np.uint8))
    concat.save(os.path.join(outdir, 'stitch', 'concat', filename))

def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference adapter', add_help=False)
    parser.add_argument('--base-model-path', type=str, help='path to base ckpt', required=True)
    parser.add_argument('--adapter-path', type=str, help='path to adapter ckpt', required=True)
    parser.add_argument('--class-idx', type=int, help='idx of the novel class', required=True)
    parser.add_argument('--split', type=int, help='how many to image split into (each dim)', default=2)
    parser.add_argument('--dataset-dir', type=str, help='path to input image to be tested', default='/disk3/steve/dataset/OpenEarthMap-FSS/testset/images')
    parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
    parser.add_argument('--stitch-width', type=int, help='width of the stitching', default=4)
    parser.add_argument('--outdir', type=str, help='path to output directory', default='./')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    seggpt_model = seggpt_vit_large_patch16_input896x448()
    ckpt = T.load(args.base_model_path, map_location='cpu')
    seggpt_model.load_state_dict(ckpt['model_state_dict'])
    print('SegGPT base model loaded')
    model = AdapterSegGPT(seggpt_model)
    adapter_ckpt = T.load(args.adapter_path, map_location='cpu')
    model.image_tensor.data = adapter_ckpt['model_state_dict']['image_tensor']
    model.mask_tensor.data = adapter_ckpt['model_state_dict']['mask_tensor']
    print('Adapter loaded')

    model = model.to(args.device)
    model.eval()

    for file in tqdm(os.listdir(args.dataset_dir)):
        inference_image_with_crop(model, 'cuda', os.path.join(args.dataset_dir, file), args.class_idx, outdir=args.outdir, split=args.split)
        tgt_path = os.path.join(args.outdir, 'color', file.replace('.tif', '.png'))
        lbl_path = os.path.join(args.outdir, 'label', file.replace('.tif', '.png'))
        if args.split == 2:
            inference_stitch(model, 'cuda', os.path.join(args.dataset_dir, file), args.class_idx, tgt_path, lbl_path, args.outdir, split=args.split, width=args.stitch_width)