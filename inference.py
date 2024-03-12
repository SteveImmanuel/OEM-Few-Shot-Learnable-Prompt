import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import os
import argparse
import torch
import numpy as np
import models_seggpt
import torch.nn.functional as F
from PIL import Image
from utils import cmap_to_lbl

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

color_map = np.array([
    (0, 0, 0),      # Background (e.g., sky)
    (34, 139, 34),  # Tree (Forest Green)
    (0, 255, 127),  # Rangeland (Chartreuse)
    (0, 255, 36),   # Bareland (Green)
    (244, 164, 96), # Agricultural Land Type 1 (Sandy Brown)
    (255, 255, 255),# Road Type 1 (White)
    (0, 191, 255),  # Sea, Lake, & Pond (Deep Sky Blue)
    (255, 0, 0),    # Building Type 1 (Red)
    # (218, 165, 32), # Road Type 2 (Goldenrod)
    # (65, 105, 225), # River (Royal Blue)
    # (0, 255, 127),  # Boat & Ship (Spring Green)
    # (107, 142, 35), # Agricultural Land Type 2 (Olive Drab)
    # (240, 230, 140),# (Add meaningful label) (Khaki)
    # (128, 0, 128),  # (Add meaningful label) (Purple)
    # (255, 20, 147)  # (Add meaningful label) (Deep Pink)
])

def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

@torch.no_grad()
def run_one_image(img, tgt, model, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    if model.seg_type == 'instance':
        seg_type = torch.ones([valid.shape[0], 1])
    else:
        seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output

def inference_image(model, device, img_path, img2_paths, tgt2_paths, out_path):
    res, hres = 448, 448

    image = Image.open(img_path).convert("RGB")
    input_image = np.array(image)
    size = image.size
    image = np.array(image.resize((res, hres))) / 255.

    image_batch, target_batch = [], []
    for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path).convert("RGB")
        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)
    
        assert img.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        assert tgt.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        image_batch.append(img)
        target_batch.append(tgt)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)
    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    output = run_one_image(img, tgt, model, device)
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), 
        size=[size[1], size[0]], 
        mode='nearest',
    ).permute(0, 2, 3, 1)
    print(output.shape)
    output, label = cmap_to_lbl(output, torch.tensor(color_map, device=output.device, dtype=output.dtype).unsqueeze(0))

    output = output[0].numpy()
    output = Image.fromarray((output).astype(np.uint8))
    output.save(out_path)



if __name__ == '__main__':
    args = get_args_parser()

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    assert args.input_image
    assert args.prompt_image is not None and args.prompt_target is not None

    img_name = os.path.basename(args.input_image)
    out_path = os.path.join(args.output_dir, "output_" + '.'.join(img_name.split('.')[:-1]) + '.png')

    inference_image(model, device, args.input_image, args.prompt_image, args.prompt_target, out_path)
    

    print('Finished.')
"""
python seggpt_inference.py --ckpt_path /home/steve/SegGPT-FineTune/logs/1710148218/weights/epoch15_loss0.7601_metric0.0000.pt \
--input_image /disk3/steve/dataset/OpenEarthMap-FSS/valset/images/accra_29.tif \
--prompt_image /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images/accra_8.tif /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images/accra_27.tif /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images/accra_31.tif /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images/accra_37.tif \
--prompt_target /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color/5/accra_8.png /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color/6/accra_27.png /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color/5/accra_31.png /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color/6/accra_37.png \
--output_dir ./
"""