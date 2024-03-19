import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import os, json
import argparse
import torch
import numpy as np, cv2
from tqdm import tqdm
import models_seggpt
import torch.nn.functional as F
from PIL import Image
from utils import cmap_to_lbl

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# color_map = np.array([
#     # (0, 0, 0),      # Background (e.g., sky)
#     (34, 139, 34),  # Tree (Forest Green)
#     (0, 255, 127),  # Rangeland (Chartreuse)
#     (0, 255, 36),   # Bareland (Green)
#     (244, 164, 96), # Agricultural Land Type 1 (Sandy Brown)
#     (255, 255, 255),# Road Type 1 (White)
#     (0, 191, 255),  # Sea, Lake, & Pond (Deep Sky Blue)
#     (255, 0, 0),    # Building Type 1 (Red)
#     # (218, 165, 32), # Road Type 2 (Goldenrod)
#     # (65, 105, 225), # River (Royal Blue)
#     # (0, 255, 127),  # Boat & Ship (Spring Green)
#     # (107, 142, 35), # Agricultural Land Type 2 (Olive Drab)
#     # (240, 230, 140),# (Add meaningful label) (Khaki)
#     # (128, 0, 128),  # (Add meaningful label) (Purple)
#     # (255, 20, 147)  # (Add meaningful label) (Deep Pink)
# ])

color_map = np.array([
    (0, 0,  0),
    (40, 130,  72),
    (255, 237, 2),
    (222, 173,  100),
    (215,  22, 194),
    (255, 255, 255),
    (59,  17, 243),
    (114,   6,  39),
])

# color_map = np.array([
#     (0, 0,  0),
#     (40, 130,  72),
#     (255, 237, 2),
#     (222, 173,  100),
#     (255,  0, 0),
#     (255, 255, 255),
#     (59,  17, 243),
#     (114,   6,  39),
# ])

def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--mapping', type=str, help='path to mapping of query and prompt list',
                        default="inference_mapping.json")
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='semantic')
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
def run_one_image(img, tgt, model, device, mask=None):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    if mask is None:
        bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
        bool_masked_pos[model.patch_embed.num_patches//2:] = 1
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    else:
        bool_masked_pos = torch.tensor(mask).unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    # print('outmodel', y.shape)
    output = y[0, y.shape[1]//2:, :, :] 
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)


    mask = mask[:, :, None].repeat(1, 1, model.patch_size**2 * 3)
    mask = model.unpatchify(mask)
    # print('m', mask.shape)
    mask = mask.permute(0, 2, 3, 1)
    mask = mask[0, mask.shape[1]//2:, :, :]
    mask = mask.cpu().float()
    # temp_img = x[0].permute(1, 2, 0)
    # print(temp_img.shape)
    # masked_label = (1 - mask) * temp_img  + mask * torch.randn_like(mask)
    # masked_label = torch.clip((masked_label * imagenet_std + imagenet_mean) * 255, 0, 255)
    # masked_label = masked_label.cpu().numpy()
    # print(masked_label.shape)
    # timg = Image.fromarray((masked_label).astype(np.uint8))
    # timg.save('temp_test.png')


    return output, mask

def inference_image_with_crop(model, device, img_path, img2_paths, tgt2_paths, out_path, store_dir=False, split=2):
    res, hres = 448, 448

    full_image = Image.open(img_path).convert("RGB").resize((1024, 1024))
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

            image_batch, target_batch = [], []
            for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
                full_img2 = Image.open(img2_path).convert("RGB").resize((1024, 1024))
                full_tgt2 = Image.open(tgt2_path).convert("RGB").resize((1024, 1024), Image.NEAREST)

                for i_row in range(split):
                    for i_col in range(split):
                        img2 = full_img2.crop((i_row * row_size, i_col * col_size, (i_row + 1) * row_size, (i_col + 1) * col_size))
                        tgt2 = full_tgt2.crop((i_row * row_size, i_col * col_size, (i_row + 1) * row_size, (i_col + 1) * col_size))

                        img2 = img2.resize((res, hres))
                        img2 = np.array(img2) / 255.

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
            output, _ = run_one_image(img, tgt, model, device)
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


    concat_img = np.concatenate((final_out_image, final_out_color), axis=1)
    final_out_color = Image.fromarray((final_out_color).astype(np.uint8))

    if store_dir:
        dirname, filename = os.path.dirname(out_path), os.path.basename(out_path)
        color_dir = os.path.join(dirname, "color"); os.makedirs(color_dir, exist_ok=True)
        final_out_color.save(os.path.join(color_dir, filename))

        final_out_label = Image.fromarray((final_out_label).astype(np.uint8))
        label_dir = os.path.join(dirname, "label"); os.makedirs(label_dir, exist_ok=True)
        final_out_label.save(os.path.join(label_dir, filename))

        concat_img = Image.fromarray((concat_img).astype(np.uint8))
        concat_dir = os.path.join(dirname, "concat"); os.makedirs(concat_dir, exist_ok=True)
        concat_img.save(os.path.join(concat_dir, filename))
    else:
        final_out_color.save(out_path)


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

def inference_stitch(model, device, img_path, tgt_path, lbl_path, img2_paths, tgt2_paths, out_path, store_dir=False, split=2, width=4):
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

    for crop_param, stitch_type in crop_params:
        j1, i1, j2, i2 = crop_param
        assert j2 - j1 == col_size and i2 - i1 == row_size

        cropped_image = full_image.crop(crop_param).resize((res, hres))
        cropped_tgt = full_tgt.crop(crop_param).resize((res, hres), Image.NEAREST)
        cropped_image = np.array(cropped_image.resize((res, hres))) / 255.
        cropped_tgt = np.array(cropped_tgt) / 255.

        image_batch, target_batch = [], []
        for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
            full_img2 = Image.open(img2_path).convert('RGB').resize((1024, 1024))
            full_tgt2 = Image.open(tgt2_path).convert('RGB').resize((1024, 1024), Image.NEAREST)

            for i_row in range(split):
                for i_col in range(split):
                    img2 = full_img2.crop((i_row * row_size, i_col * col_size, (i_row + 1) * row_size, (i_col + 1) * col_size))
                    tgt2 = full_tgt2.crop((i_row * row_size, i_col * col_size, (i_row + 1) * row_size, (i_col + 1) * col_size))

                    img2 = img2.resize((res, hres))
                    img2 = np.array(img2) / 255.

                    tgt2 = tgt2.resize((res, hres), Image.NEAREST)
                    tgt2 = np.array(tgt2) / 255.

                    tgt = cropped_tgt
                    tgt = np.concatenate((tgt2, tgt), axis=0)
                    img = np.concatenate((img2, cropped_image), axis=0)

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
        final_out_label[i1:i2, j1:j2] = label * mask[:, :, 0] + final_out_label[i1:i2, j1:j2] * (1 - mask[:, :, 0])

    final_out_color = Image.fromarray((final_out_color).astype(np.uint8))

    if store_dir:
        dirname, filename = os.path.dirname(out_path), os.path.basename(out_path)
        dirname = f'{dirname}_stitch'
        color_dir = os.path.join(dirname, "color"); os.makedirs(color_dir, exist_ok=True)
        final_out_color.save(os.path.join(color_dir, filename))

        final_out_label = Image.fromarray((final_out_label).astype(np.uint8))
        label_dir = os.path.join(dirname, "label"); os.makedirs(label_dir, exist_ok=True)
        final_out_label.save(os.path.join(label_dir, filename))
    else:
        final_out_color.save(out_path)


def inference_image(model, device, img_path, img2_paths, tgt2_paths, out_path, store_dir=False):
    res, hres = 448, 448

    image = Image.open(img_path).convert("RGB")
    input_image = np.array(image)
    if store_dir: size = (1024,1024)
    else: size = image.size
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
    output, _ = run_one_image(img, tgt, model, device)
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), 
        size=[size[1], size[0]], 
        mode='nearest',
    ).permute(0, 2, 3, 1)
    output, label = cmap_to_lbl(output, torch.tensor(color_map, device=output.device, dtype=output.dtype).unsqueeze(0))

    output = output[0].numpy()
    output = np.concatenate((cv2.resize(input_image, (output.shape[0], output.shape[1])), output), axis=1)
    output = Image.fromarray((output).astype(np.uint8))
    if store_dir:
        dirname, filename = os.path.dirname(out_path), os.path.basename(out_path)
        color_dir = os.path.join(dirname, "color"); os.makedirs(color_dir, exist_ok=True)
        output.save(os.path.join(color_dir, filename))

        label = label[0].numpy()
        label = Image.fromarray((label).astype(np.uint8))
        label_dir = os.path.join(dirname, "label"); os.makedirs(label_dir, exist_ok=True)
        label.save(os.path.join(label_dir, filename))
    else:
        output.save(out_path)

def run_eval(args, model):
    mapping = json.load(open(args.mapping))
    prompt_folder, val_folder = '/disk3/steve/dataset/OpenEarthMap-FSS/trainset/images', '/disk3/steve/dataset/OpenEarthMap-FSS/testset/images'
    prompt_label_color_folder = '/disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color'
    for input_image in tqdm(mapping):
        input = os.path.join(val_folder, input_image)
        top_k = 2
        prompt = [os.path.join(prompt_folder, file) for file in mapping[input_image][:top_k]]
        prompt_target = [os.path.join(prompt_label_color_folder, file.replace('.tif', '.png')) for file in mapping[input_image][:top_k]]
        out_path = os.path.join(args.output_dir, input_image.replace('.tif', '.png'))
        inference_image_with_crop(model, device, input, prompt, prompt_target, out_path, store_dir=True, split=2)
        # inference_image(model, device, input, prompt, prompt_target, out_path, store_dir=True)
        # print('inference crop done')
        tgt_path = os.path.join(args.output_dir, 'color', input_image.replace('.tif', '.png'))
        lbl_path = os.path.join(args.output_dir, 'label', input_image.replace('.tif', '.png'))
        inference_stitch(model, device, input, tgt_path, lbl_path, prompt, prompt_target, out_path, store_dir=True, split=2, width=5)
        # print('stitch done')


if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    if not args.input_image:
        run_eval(args, model)
    else:
        assert args.input_image
        assert args.prompt_image is not None and args.prompt_target is not None

        img_name = os.path.basename(args.input_image)
        out_path = os.path.join(args.output_dir, "output_" + '.'.join(img_name.split('.')[:-1]) + '.png')

        inference_image(model, device, args.input_image, args.prompt_image, args.prompt_target, out_path)
    
    print('Finished.')

"""
python inference.py --ckpt_path /home/steve/SegGPT-FineTune/logs/1710148218/weights/epoch15_loss0.7601_metric0.0000.pt --output_dir submission

python seggpt_inference.py --ckpt_path ../../../tuning.pt --input_image /home/steve/Datasets/OpenEarthMap-FSS/valset/images/tonga_64.tif --prompt_image /home/steve/Datasets/OpenEarthMap-FSS/valset/images/christchurch_39.tif /home/steve/Datasets/OpenEarthMap-FSS/valset/images/sechura_37.tif /home/steve/Datasets/OpenEarthMap-FSS/valset/images/kitsap_22.tif /home/steve/Datasets/OpenEarthMap-FSS/valset/images/duesseldorf_15.tif /home/steve/Datasets/OpenEarthMap-FSS/valset/images/sechura_11.tif --prompt_target /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/christchurch_39.png /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/sechura_37.png /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/kitsap_22.png /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/duesseldorf_15.png /home/steve/Datasets/OpenEarthMap-FSS/valset/labels_color/sechura_11.png --output_dir tuning
"""