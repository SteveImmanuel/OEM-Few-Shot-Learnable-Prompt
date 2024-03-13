from typing import Iterable, Tuple
import os

from tqdm import tqdm
import numpy as np
import torch

from data import OEMDataset

class OEMNovelMixDataset(OEMDataset):
    def __init__(
        self, 
        root:str, 
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        max_classes: int = 11, # including background
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        is_train: bool = True,
        mapping_file: str = "cache/train_mapping.npy",
    ):
        super().__init__(root, mean, std, resize, max_classes, patch_size, mask_ratio, is_train)
        if not self.is_train:
            self.color_palette = self._generate_color_palette()
        self.pairs = np.load(mapping_file, allow_pickle=True)

    def _preload_dataset(self):
        self.images = []
        self.labels = []
        self.unique_classes = []
        self.name2idx = {}
        idx = 0
        for img_path, label_path in tqdm(self.paths, desc='Caching images and labels'):
            img = self._load_img(img_path)
            label = self._load_lbl(label_path)
            self.images.append(img)
            self.labels.append(label)
            self.unique_classes.append(set(np.unique(label)))
            self.name2idx[os.path.basename(img_path)] = idx
            idx += 1
    
    def _generate_pairs(self): #here
        return

    def _filter_pairs(self): 
        return
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        label_file, all_classes, base_classes, base_file, novel_classes, novel_files = self.pairs[idx]
        # pair_idx1, pair_idx2 = self.pairs[idx] # pair_idx1 = base_file, pair_idx2 = label

        if self.is_train:
            color_palette = self._generate_color_palette()
        else:
            color_palette = self.color_palette

        label_idx = self.name2idx[label_file]
        img2, ori_label2 = self.images[label_idx], self.labels[label_idx]

        imgs, labels, masks, valids, ori_labels = [], [], [], [], []
        for query_file, allowed_classes in zip([base_file] + novel_files, [base_classes] + [set([i]) for i in novel_classes]):
            query_idx = self.name2idx[query_file]
            img1, ori_label1 = self.images[query_idx], self.labels[query_idx]
            ori_label_new = np.zeros_like(ori_label1)
            for allowed_class in allowed_classes:
                ori_label_new[ori_label1 == allowed_class] = allowed_class
            ori_label1 = ori_label_new

            label1 = self._lbl_random_color(ori_label1, color_palette)
            label2 = self._lbl_random_color(ori_label2, color_palette)

            img, label, ori_label = self._augment([img1, img2], [label1, label2], [ori_label1, ori_label2])
            img = np.concatenate(img, axis=0)
            label = np.concatenate(label, axis=0)
            ori_label = np.concatenate(ori_label, axis=0)
            
            img = self._to_img_tensor(img)
            label = self._to_img_tensor(label)
            ori_label = torch.FloatTensor(ori_label)
            
            mask = self._generate_mask((img.shape[1], img.shape[2]), is_half=True)
            valid = torch.ones_like(label)
            
            imgs.append(img)
            labels.append(label)
            masks.append(mask)
            valids.append(valid)
            ori_labels.append(ori_label)

        seg_type = torch.zeros([1+len(novel_files), 1])
        color_palette = torch.FloatTensor(color_palette)
        imgs, labels, masks = torch.stack(imgs), torch.stack(labels), torch.stack(masks)
        valids, ori_labels = torch.stack(valids), torch.stack(ori_labels)
        return imgs, labels, masks, valids, seg_type, ori_labels, color_palette

if __name__ == '__main__':
    dataset = OEMNovelMixDataset('/disk3/steve/dataset/OpenEarthMap', is_train=True)