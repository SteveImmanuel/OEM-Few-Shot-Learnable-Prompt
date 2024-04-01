import os
import os.path as osp
import shutil
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description='This script prepares the dataset for training')
    parser.add_argument('--dataset-dir', type=str, required=True, help='path to val/test dataset')
    parser.add_argument('--convert-color', action='store_true')
    return parser.parse_args()

def prepare(dataset_dir: str):
    assert osp.exists(osp.join(dataset_dir, 'images')), f'Image dir does not exist'
    assert osp.exists(osp.join(dataset_dir, 'labels')), f'Label dir does not exist'

    for file in tqdm(os.listdir(osp.join(dataset_dir, 'labels'))):
        label_path = osp.join(dataset_dir, 'labels', file)
        image_path = osp.join(dataset_dir, 'images', file)

        label = np.array(Image.open(label_path).resize((1024, 1024), Image.NEAREST))
        for i in range(8, 12):
            if i not in np.unique(label):
                continue
            else:
                os.makedirs(osp.join(dataset_dir, str(i), 'images'), exist_ok=True)
                os.makedirs(osp.join(dataset_dir, str(i), 'labels'), exist_ok=True)

                label[label == i] = 1
                label = Image.fromarray(label.astype(np.uint8))
                label.save(osp.join(dataset_dir, str(i), 'labels', file))
                os.remove(label_path)
                shutil.move(image_path, osp.join(dataset_dir, str(i), 'images', file))
                break

    os.rmdir(osp.join(dataset_dir, 'labels'))

def convert_label(label: np.ndarray, color_palette: np.ndarray):
    result = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    N = len(color_palette)
    for i in range(N):
        result[label == i] = color_palette[i]
    return result

def process_label(dataset_dir: str):
    COLOR_MAP = np.array([
        (0, 0,  0), # Background
        (255, 0,  0), # Tree
        (0, 255, 255), # Rangeland
        (0, 255,  0), # Bareland
        (255, 255, 0), # Agric land type 1
        (0,  0, 255), # Road type 1
        (255,  255, 255), # Sea, lake, & pond
        (255,   0,  255), # Building type 1
    ])

    for file in tqdm(os.listdir(osp.join(dataset_dir, 'labels'))):
        label_path = osp.join(dataset_dir, 'labels', file)
        lbl = Image.open(label_path).convert('L')
        lbl = np.array(lbl).astype(np.uint8)

        lbl = convert_label(lbl, COLOR_MAP)
        lbl = Image.fromarray(lbl)
        os.makedirs(osp.join(dataset_dir, 'labels_color'), exist_ok=True)
        lbl.save(osp.join(dataset_dir, 'labels_color', file.replace('.tif', '.png')))

if __name__ == '__main__':
    args = get_args()
    if args.convert_color:
        process_label(args.dataset_dir)
    else:
        prepare(args.dataset_dir)
