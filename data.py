import torch
import os
import os.path as osp
import numpy as np
import imgaug.augmenters as iaa
from utils import get_logger
from typing import Iterable, Tuple, List
from PIL import Image
from itertools import combinations

class OEMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root:str, 
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        max_classes: int = 11,
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        half_mask_ratio: float = 0.2,
    ):
        super().__init__()
        assert osp.exists(osp.join(root, 'images')), f'Path {root}/images does not exist'
        assert osp.exists(osp.join(root, 'labels')), f'Path {root}/labels does not exist'
        
        self.root = root
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.resize = resize
        self.max_classes = max_classes
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.half_mask_ratio = half_mask_ratio
        self.logger = get_logger(__class__.__name__, 0) # TODO: Bug, every process will log
        
        self.paths = []
        for path in os.listdir(osp.join(self.root, 'images')):
            img_path = osp.join(self.root, 'images', path)
            label_path = osp.join(self.root, 'labels', path)
            if not osp.exists(label_path):
                self.logger.warn(f'Skipping label path {label_path} as it does not exist')

            self.paths.append((img_path, label_path))
        
        self._generate_pairs()
        self._init_augmentation()

    def _generate_pairs(self):
        indices = np.arange(len(self.paths))
        self.pairs = list(combinations(indices, 2))

    def _load_img_lbl(self, img_path, label_path):
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        if self.resize is not None:
            img = img.resize(self.resize)
            label = label.resize(self.resize)
        img = np.array(img).astype(np.uint8)
        label = np.array(label).astype(np.uint8)
        return img, label

    def _generate_color_palette(self):
        return np.random.randint(0, 256, (self.max_classes, 3))
    
    def _init_augmentation(self):
        self.augment_all = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ])
        self.augment_img = iaa.Sequential([ # does not change relative positions
            iaa.GaussianBlur((0, 0.05))
        ])
    
    def _augment(self, img: List[np.ndarray], label: List[np.ndarray]):
        aug_all = self.augment_all.to_deterministic()
        all = np.concatenate([img, label], axis=0)
        res = [aug_all.augment_image(x) for x in all]
        img, label = res[:len(img)], res[len(img):]
        img = self.augment_img.augment_images(img)
        return img, label

    def _lbl_random_color(self, label: np.ndarray, color_palette: np.ndarray):
        result = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for i in range(self.max_classes):
            result[label == (i + 1)] = color_palette[i] # 0 is reserved for background
        return result

    def _to_tensor(self, arr: np.ndarray):
        arr = arr / 255.0
        arr = arr - self.mean
        arr = arr / self.std
        res = torch.FloatTensor(arr)
        res = torch.einsum('hwc->chw', res)
        return res
    
    def _generate_mask(self, img_shape: Tuple[int, int]):
        total_patch = (img_shape[0] // self.patch_size[0]) * (img_shape[1] // self.patch_size[1])
        if np.random.rand() < self.half_mask_ratio:
            mask = torch.zeros(total_patch, dtype=torch.float32)
            mask[total_patch//2:] = 1
        else:
            total_zeros = int(total_patch * self.mask_ratio)
            shuffle_idx = torch.randperm(total_patch)
            mask = torch.FloatTensor([0] * total_zeros + [1] * (total_patch - total_zeros))[shuffle_idx]
        return mask

    def __getitem__(self, idx):
        pair_idx1, pair_idx2 = self.pairs[idx]
        img1, label1 = self._load_img_lbl(*self.paths[pair_idx1])
        img2, label2 = self._load_img_lbl(*self.paths[pair_idx2])
        if np.random.rand() > 0.5: # swap pair
            img1, img2 = img2, img1
            label1, label2 = label2, label1

        color_palette = self._generate_color_palette()
        label1 = self._lbl_random_color(label1, color_palette)
        label2 = self._lbl_random_color(label2, color_palette)

        img, label = self._augment([img1, img2], [label1, label2])
        img = np.concatenate(img, axis=0)
        label = np.concatenate(label, axis=0)
        img = self._to_tensor(img)
        label = self._to_tensor(label)
        
        valid = torch.ones_like(label)
        mask = self._generate_mask((img.shape[1], img.shape[2]))
        seg_type = torch.zeros([1])
        return img, label, mask, valid, seg_type

    def __len__(self):
        return len(self.pairs)

if __name__ == '__main__':
    dataset = OEMDataset('/home/steve/Datasets/OpenEarthMap-FSS/trainset', half_mask_ratio=0)
    # img, label = dataset[0]
    img, label, mask, valid, color_palette = dataset[0]
    img = img.cpu().numpy()
    label = label.cpu().numpy()
    img = (((img * dataset.std) + dataset.mean) * 255).astype(np.uint8)
    lbl = (((label * dataset.std) + dataset.mean) * 255).astype(np.uint8)
    print(img.shape, label.shape)
    print(mask)