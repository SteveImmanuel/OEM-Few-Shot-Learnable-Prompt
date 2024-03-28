import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import os, json
from collections import defaultdict
import argparse
import torch
import numpy as np, cv2
from tqdm import tqdm
from Painter.SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448
import torch.nn.functional as F
from PIL import Image
from utils import cmap_to_lbl
from transformers import AutoProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

EMBED_MAPPING = np.load("cache/multiscale_mapping.npy", allow_pickle=True).item()

color_map = np.array([
    # (0, 0, 0),      # Background (e.g., sky)
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

color_map = np.array([
    (0, 0,  0),
    (40, 130,  72),
    (255, 237, 2),
    (222, 173,  100),
    (215,  22, 194),
    (255, 255, 255),
    (59,  17, 243),
    (114,   6,  39),
    [137, 235, 240],
    [158, 164, 166],
    [250, 235, 123],
    (140, 230, 240),
    (128, 0, 128),
])

EMBED_MAPPING = np.load("cache/multiscale_mapping.npy", allow_pickle=True).item()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()


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
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    return parser.parse_args()


def prepare_model(model_path):
    model = seggpt_vit_large_patch16_input896x448()
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print('Checkpoint loaded')

    model = model.to(args.device)
    model.eval()
    return model

def canny_edge_detection(img, low_threshold=150, high_threshold=200):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return edges

def prepare_divider(canny):
    h, w = canny.shape
    divider = defaultdict(list)
    cand = [[w, 0,0]]
    while len(cand):
        size, x, y = cand.pop()
        patch = canny[x:x+size,y:y+size]
        if np.sum(patch) < 1e6:
            divider[size].append([x, y])
        else:
            size = size // 2
            cand.append([size, x, y])
            cand.append([size, x+size, y])
            cand.append([size, x, y+size])
            cand.append([size, x+size, y+size])
    return divider

@torch.no_grad()
def run_one_image(img, tgt, model, device, mask=None):
    x = torch.tensor(img)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
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

    output = y[0, y.shape[1]//2:, :, :] 
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)

    mask = mask[:, :, None].repeat(1, 1, model.patch_size**2 * 3)
    mask = model.unpatchify(mask)
    mask = mask.permute(0, 2, 3, 1)
    mask = mask[0, mask.shape[1]//2:, :, :]
    mask = mask.cpu().float()
    return output, mask

def inference_image(model, device, img_path, img2_paths, tgt2_paths, out_path, store_dir=False):
    res, hres = 448, 448
    PROMPT_SIZE = 2
    
    print(img_path, img2_paths, tgt2_paths, out_path)
    image = cv2.imread(img_path)[:,:,::-1]
    if store_dir: size = (1024,1024)
    else: size = image.size
    image = np.array(cv2.resize(image, (1024,1024)))
    canny_image = canny_edge_detection(image)
    base_image = np.array(image) / 255.
    divider = prepare_divider(canny_image)
    
    # caching image
    imgs2, tgts2 = {}, {}
    for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
        bn = os.path.basename(img2_path)
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((1024,1024))
        img2 = np.array(img2) / 255.
        imgs2[bn] = img2

        tgt2 = Image.open(tgt2_path).convert("RGB")
        tgt2 = tgt2.resize((1024,1024), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.
        tgts2[bn] = tgt2
    
    final_output, final_label = np.zeros_like(base_image), np.zeros_like(canny_image)
    for patch_size in divider:
        # preparing embed
        names, embeds = [], []
        scale = 1024 // patch_size
        for file in img2_paths:
            bn = os.path.basename(file)
            names.append(EMBED_MAPPING[bn][scale]["names"])
            embeds.append(EMBED_MAPPING[bn][scale]["embeds"])
        names = np.concatenate(names, axis=0)
        embeds = np.concatenate(embeds, axis=0)
        
        # inference per patch
        for x, y in divider[patch_size]:
            image = base_image[x:x+patch_size, y:y+patch_size, :]
            image = cv2.resize(image, (res, hres))
            
            # finding close match with embeds
            inputs = clip_processor(images=image, return_tensors="pt")
            image_features = clip_model.get_image_features(**inputs)
            similarity = cosine_similarity(image_features[None,0,:].detach().numpy(), embeds)
            idx_order = np.argsort(similarity)[0,::-1]
            
            image_batch, target_batch = [], []
            for idx in idx_order[:PROMPT_SIZE]:
                data = names[idx].split("_")
                fn, xx, yy = data[0] + "_" + data[1], int(data[2]), int(data[3])
                img2 = imgs2[fn][xx:xx+patch_size, yy:yy+patch_size, :]
                img2 = cv2.resize(img2, (res, hres))
                tgt2 = tgts2[fn][xx:xx+patch_size, yy:yy+patch_size, :]
                tgt2 = cv2.resize(tgt2, (res, hres))

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
            print(img.shape, tgt.shape)
            output, _ = run_one_image(img, tgt, model, device)
            output = F.interpolate(
                output[None, ...].permute(0, 3, 1, 2), 
                size=[patch_size, patch_size], 
                mode='nearest',
            ).permute(0, 2, 3, 1)
            output, label = cmap_to_lbl(output, torch.tensor(color_map, device=output.device, dtype=output.dtype).unsqueeze(0))
            final_output[x:x+patch_size, y:y+patch_size, :] = output[0].numpy()
            final_label[x:x+patch_size, y:y+patch_size] = label[0].numpy()
        
    output = np.concatenate((base_image, final_output), axis=1)
    output = Image.fromarray((output).astype(np.uint8))
    if store_dir:
        dirname, filename = os.path.dirname(out_path), os.path.basename(out_path)
        color_dir = os.path.join(dirname, "color"); os.makedirs(color_dir, exist_ok=True)
        output.save(os.path.join(color_dir, filename))

        label = Image.fromarray((final_label).astype(np.uint8))
        label_dir = os.path.join(dirname, "label"); os.makedirs(label_dir, exist_ok=True)
        label.save(os.path.join(label_dir, filename))
    else:
        output.save(out_path)
              
def run_eval(args, model):
    mapping = json.load(open(args.mapping))
    train_folder, val_folder = "./dataset/trainset/images", "./dataset/testset/images"
    train_color_folder = "./dataset/trainset/labels_colored"
    for input_image in tqdm(mapping):
        input = os.path.join(val_folder, input_image)
        prompt = [os.path.join(train_folder, file) for file in mapping[input_image]]
        prompt_target = [os.path.join(train_color_folder, file.replace('.tif', '.png')) for file in mapping[input_image]]
        out_path = os.path.join(args.output_dir, input_image.replace('.tif', '.png'))
        inference_image(model, device, input, prompt, prompt_target, out_path, store_dir=True)
    return

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

python seggpt_inference.py --ckpt_path /home/steve/SegGPT-FineTune/logs/1710148218/weights/epoch15_loss0.7601_metric0.0000.pt \
--input_image /disk3/steve/dataset/OpenEarthMap-FSS/valset/images/accra_29.tif \
--prompt_image /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images/accra_8.tif /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images/accra_27.tif /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images/accra_31.tif /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images/accra_37.tif \
--prompt_target /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color/5/accra_8.png /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color/6/accra_27.png /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color/5/accra_31.png /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels_color/6/accra_37.png \
--output_dir ./
"""