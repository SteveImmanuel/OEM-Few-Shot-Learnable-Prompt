import torch
import os
import os.path as osp
import numpy as np
import imgaug.augmenters as iaa
from utils import get_logger
from typing import Iterable, Tuple, List
from PIL import Image
from itertools import combinations, permutations
from tqdm import tqdm

class OEMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root:str, 
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        max_classes: int = 12, # including background
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        is_train: bool = True,
    ):
        super().__init__()
        assert osp.exists(osp.join(root, 'images')), f'Path {root}/images does not exist'
        assert osp.exists(osp.join(root, 'labels')), f'Path {root}/labels does not exist'
        
        self.root = root
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.resize = resize
        self.is_train = is_train
        self.max_classes = max_classes
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.logger = get_logger(__class__.__name__, 0) # TODO: Bug, every process will log
        
        self.paths = []
        for path in os.listdir(osp.join(self.root, 'images')):
            img_path = osp.join(self.root, 'images', path)
            label_path = osp.join(self.root, 'labels', path)
            if not osp.exists(label_path):
                # self.logger.warn(f'Skipping label path {label_path} as it does not exist')
                continue
            self.paths.append((img_path, label_path))
        
        self._preload_dataset()
        self._generate_pairs()
        self._init_augmentation()
        self._filter_pairs()

    def _preload_dataset(self):
        self.images = []
        self.labels = []
        self.unique_classes = []
        for img_path, label_path in tqdm(self.paths, desc='Caching images and labels'):
            img = self._load_img(img_path)
            label = self._load_lbl(label_path)
            self.images.append(img)
            self.labels.append(label)
            self.unique_classes.append(set(np.unique(label)))

    def _generate_pairs(self):
        indices = np.arange(len(self.paths))
        if self.is_train:
            self.pairs = list(combinations(indices, 2))
        else:
            self.pairs = list(permutations(indices, 2))
    
    def _filter_pairs(self):
        self.same_class_pairs = []
        self.diff_class_pairs = []
        for pair in tqdm(self.pairs, desc='Filtering pairs'):
            len_intersect = self.unique_classes[pair[0]].intersection(self.unique_classes[pair[1]])
            len_union = self.unique_classes[pair[0]].union(self.unique_classes[pair[1]])
            if len_intersect == len_union:
                self.same_class_pairs.append(pair)
            else:
                self.diff_class_pairs.append(pair)
        np.random.shuffle(self.same_class_pairs)
        np.random.shuffle(self.diff_class_pairs)

    def _load_img(self, path):
        img = Image.open(path).convert('RGB')
        if self.resize is not None:
            img = img.resize(self.resize)
        img = np.array(img).astype(np.uint8)
        return img
    
    def _load_lbl(self, path):
        label = Image.open(path).convert('L')
        if self.resize is not None:
            label = label.resize(self.resize, Image.NEAREST)
        label = np.array(label).astype(np.uint8)
        return label

    def _load_img_lbl(self, img_path, label_path):
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        if self.resize is not None:
            img = img.resize(self.resize)
            label = label.resize(self.resize, Image.NEAREST)
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
    
    def _augment(self, img: List[np.ndarray], label: List[np.ndarray], ori_label: List[np.ndarray]):
        aug_all = self.augment_all.to_deterministic()
        all = np.concatenate([img, label], axis=0)
        res = [aug_all.augment_image(x) for x in all]
        ori_label = [aug_all.augment_image(x) for x in ori_label]
        img, label = res[:len(img)], res[len(img):]
        img = self.augment_img.augment_images(img)
        return img, label, ori_label

    def _lbl_random_color(self, label: np.ndarray, color_palette: np.ndarray):
        result = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for i in range(self.max_classes):
            result[label == i] = color_palette[i]
        return result

    def _to_img_tensor(self, arr: np.ndarray):
        arr = arr / 255.0
        arr = arr - self.mean
        arr = arr / self.std
        res = torch.FloatTensor(arr)
        res = torch.einsum('hwc->chw', res)
        return res
    
    def _generate_mask(self, img_shape: Tuple[int, int], is_half: bool = False):
        # 1 means masked, 0 means not masked
        total_patch = (img_shape[0] // self.patch_size[0]) * (img_shape[1] // self.patch_size[1])
        if is_half:
            mask = torch.zeros(total_patch, dtype=torch.float32)
            mask[total_patch//2:] = 1
        else:
            total_ones = int(total_patch * self.mask_ratio)
            shuffle_idx = torch.randperm(total_patch)
            mask = torch.FloatTensor([0] * (total_patch - total_ones) + [1] * total_ones)[shuffle_idx]

        return mask

    def __getitem__(self, idx):
        if self.is_train:
            if idx < len(self.same_class_pairs):
                pair_idx1, pair_idx2 = self.same_class_pairs[idx]
            else:
                pair_idx1, pair_idx2 = self.diff_class_pairs[idx - len(self.same_class_pairs)]

            if np.random.rand() > 0.5: # swap pair
                pair_idx1, pair_idx2 = pair_idx2, pair_idx1
        else:
            pair_idx1, pair_idx2 = self.same_class_pairs[idx]

        img1, ori_label1 = self.images[pair_idx1], self.labels[pair_idx1]
        img2, ori_label2 = self.images[pair_idx2], self.labels[pair_idx2]

        color_palette = self._generate_color_palette()
        label1 = self._lbl_random_color(ori_label1, color_palette)
        label2 = self._lbl_random_color(ori_label2, color_palette)

        img, label, ori_label = self._augment([img1, img2], [label1, label2], [ori_label1, ori_label2])
        img = np.concatenate(img, axis=0)
        label = np.concatenate(label, axis=0)
        ori_label = np.concatenate(ori_label, axis=0)
        
        img = self._to_img_tensor(img)
        label = self._to_img_tensor(label)
        ori_label = torch.FloatTensor(ori_label)
        
        if not self.is_train:
            is_half = True
        else:
            is_half = idx < len(self.same_class_pairs)
        mask = self._generate_mask((img.shape[1], img.shape[2]), is_half)
        valid = torch.ones_like(label)
        seg_type = torch.zeros([1])
        color_palette = torch.FloatTensor(color_palette)
        return img, label, mask, valid, seg_type, ori_label, color_palette

    def __len__(self):
        if self.is_train:
            return len(self.same_class_pairs) + len(self.diff_class_pairs)
        return min(len(self.same_class_pairs), 1600)

class OEMFullDataset(OEMDataset):
    def __init__(
        self, 
        root:str, 
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        max_classes: int = 12, # including background
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        is_train: bool = True,
    ):
        super().__init__(root, mean, std, resize, max_classes, patch_size, mask_ratio, is_train)

        # reduce class index by 1 if it is not 0
        for i in range(len(self.labels)):
            l = self.labels[i]
            l[l > 0] -= 1
            self.labels[i] = l

        self.same_class_pairs = self.same_class_pairs[:len(self.diff_class_pairs)]    
    
class OEMHalfMaskDataset(OEMDataset):
    def __init__(
        self, 
        root:str, 
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        max_classes: int = 12, # including background
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        is_train: bool = True,
    ):
        super().__init__(root, mean, std, resize, max_classes, patch_size, mask_ratio, is_train)
        if not self.is_train:
            self.color_palette = self._generate_color_palette()
    
    def _generate_pairs(self):
        indices = np.arange(len(self.paths))
        self.pairs = list(combinations(indices, 2))

    def _filter_pairs(self):
        self.same_class_pairs = []
        self.diff_class_pairs = []
        for pair in tqdm(self.pairs, desc='Filtering pairs'):
            len_intersect = len(self.unique_classes[pair[0]].intersection(self.unique_classes[pair[1]]))
            len_union = len(self.unique_classes[pair[0]].union(self.unique_classes[pair[1]]))
            if len_intersect == len_union:
                self.same_class_pairs.append(pair)
            elif len_intersect / len_union >= 0.8:
                self.diff_class_pairs.append(pair)

        np.random.shuffle(self.same_class_pairs)
        np.random.shuffle(self.diff_class_pairs)
        cut_off_idx = int(0.95 * len(self.same_class_pairs))
        if self.is_train:
            self.same_class_pairs = self.same_class_pairs[:cut_off_idx]
        else:
            self.same_class_pairs = self.same_class_pairs[cut_off_idx:]
    
    def __len__(self):
        return len(self.same_class_pairs)

    def __getitem__(self, idx):
        pair_idx1, pair_idx2 = self.same_class_pairs[idx]

        if self.is_train and np.random.rand() > 0.5: # swap pair
            pair_idx1, pair_idx2 = pair_idx2, pair_idx1

        img1, ori_label1 = self.images[pair_idx1], self.labels[pair_idx1]
        img2, ori_label2 = self.images[pair_idx2], self.labels[pair_idx2]

        if self.is_train:
            color_palette = self._generate_color_palette()
        else:
            color_palette = self.color_palette

        label1 = self._lbl_random_color(ori_label1, color_palette)
        label2 = self._lbl_random_color(ori_label2, color_palette)

        img, label, ori_label = self._augment([img1, img2], [label1, label2], [ori_label1, ori_label2])
        img = np.concatenate(img, axis=0)
        label = np.concatenate(label, axis=0)
        ori_label = np.concatenate(ori_label, axis=0)
        
        img = self._to_img_tensor(img)
        label = self._to_img_tensor(label)
        ori_label = torch.FloatTensor(ori_label)
        
        mask = self._generate_mask((img.shape[1], img.shape[2]), True)
        valid = torch.ones_like(label)
        seg_type = torch.zeros([1])
        color_palette = torch.FloatTensor(color_palette)
        return img, label, mask, valid, seg_type, ori_label, color_palette

class OEMOneClassDataset(OEMDataset):
    def __init__(
        self, 
        root:str, 
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        max_classes: int = 12, # including background
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        negative_pairs_ratio: float = 0.3,
        validation_ratio: float = 0.1,
        is_train: bool = True,
    ):
        self.negative_pairs_ratio = negative_pairs_ratio
        self.validation_ratio = validation_ratio

        super().__init__(root, mean, std, resize, max_classes, patch_size, mask_ratio, is_train)

        if not self.is_train:
            self.color_palette = np.array([[0, 0, 0], [255, 255, 255]])
        self.class_weights = 20 / (np.log(list(self.total_pixels.values())))
        self.class_weights = np.concatenate([min(self.class_weights).reshape(1), self.class_weights], axis=0)
        self.class_weights = self.class_weights / min(self.class_weights)

    def _generate_color_palette(self):
        return np.random.randint(0, 256, (2, 3)) # background and one class

    def _lbl_random_color(self, label: np.ndarray, color_palette: np.ndarray):
        result = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for i in range(2):
            result[label == i] = color_palette[i]
        return result
        
    def _preload_dataset(self):
        self.images = []
        self.groups = {i + 1: [] for i in range(self.max_classes - 1)} # excluding background
        self.total_pixels = {i + 1: 0 for i in range(self.max_classes - 1)}
        for i, (img_path, label_path) in enumerate(tqdm(self.paths, desc='Caching images and grouping labels')):
            img = self._load_img(img_path)
            label = self._load_lbl(label_path)
            self.images.append(img)

            for j in range(1, self.max_classes):
                mask = label == j
                c_pixels = mask.sum()
                if c_pixels > 0:
                    self.groups[j].append((i, mask.astype(np.uint8)))
                self.total_pixels[j] += c_pixels
    
    def _generate_pairs(self):
        self.positive_pairs = {}
        for key in self.groups:
            self.positive_pairs[key] = list(permutations(self.groups[key], 2))
            np.random.shuffle(self.positive_pairs[key])
            cutoff_idx = min(int(len(self.positive_pairs[key]) * (self.validation_ratio)), 200)
            if self.is_train:
                self.positive_pairs[key] = self.positive_pairs[key][cutoff_idx:]
            else:
                self.positive_pairs[key] = self.positive_pairs[key][:cutoff_idx]

        self.positive_pairs = [(key, *pair)  for key in self.positive_pairs for pair in self.positive_pairs[key]]
        self.negative_pairs = list(permutations(self.groups.keys(), 2))
        np.random.shuffle(self.positive_pairs)
        np.random.shuffle(self.negative_pairs)

    def __len__(self):
        len_negative_pairs = int((len(self.positive_pairs) / (1 - self.negative_pairs_ratio)) * self.negative_pairs_ratio)
        return len(self.positive_pairs) + len_negative_pairs

    def _filter_pairs(self):
        pass

    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):
            class_label = self.positive_pairs[idx][0]
            img_idx1, ori_label1 = self.positive_pairs[idx][1]
            img_idx2, ori_label2 = self.positive_pairs[idx][2]
        else:
            class_label = 0
            idx = (idx - len(self.positive_pairs)) % len(self.negative_pairs)
            idx1, idx2 = self.negative_pairs[idx]
            img_idx1, ori_label1 = self.groups[idx1][np.random.randint(len(self.groups[idx1]))]
            img_idx2, ori_label2 = self.groups[idx2][np.random.randint(len(self.groups[idx2]))]

            ori_label2 = np.zeros_like(ori_label2)
        img1 = self.images[img_idx1]
        img2 = self.images[img_idx2]

        if self.is_train:
            color_palette = self._generate_color_palette()
        else:
            color_palette = self.color_palette

        label1 = self._lbl_random_color(ori_label1, color_palette)
        label2 = self._lbl_random_color(ori_label2, color_palette)

        img, label, ori_label = self._augment([img1, img2], [label1, label2], [ori_label1, ori_label2])
        img = np.concatenate(img, axis=0)
        label = np.concatenate(label, axis=0)
        ori_label = np.concatenate(ori_label, axis=0)
        
        img = self._to_img_tensor(img)
        label = self._to_img_tensor(label)
        ori_label = torch.FloatTensor(ori_label)
        
        mask = self._generate_mask((img.shape[1], img.shape[2]), True)
        valid = torch.ones_like(label)
        seg_type = torch.zeros([1])
        color_palette = torch.FloatTensor(color_palette)
        class_weight = torch.tensor(self.class_weights[class_label])
        class_label = torch.tensor(class_label if class_label != 0 else -1) # for IoU calculation in negative pairs
        return img, label, mask, valid, seg_type, ori_label, color_palette, class_weight, class_label

class OEMOneClassSMDataset(OEMOneClassDataset):
    def __init__(
        self, 
        root:str, 
        include_class: List[int],
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        max_classes: int = 12, # including background
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        negative_pairs_ratio: float = 0.3,
        validation_ratio: float = 0.1,
        is_train: bool = True,
    ):
        self.include_class = set(include_class)
        super().__init__(root, mean, std, resize, max_classes, patch_size, mask_ratio, negative_pairs_ratio, validation_ratio, is_train)


    def _generate_pairs(self):
        self.positive_pairs = {}
        for key in self.groups:
            self.positive_pairs[key] = list(permutations(self.groups[key], 2))
            np.random.shuffle(self.positive_pairs[key])
            cutoff_idx = min(int(len(self.positive_pairs[key]) * (self.validation_ratio)), 200)
            if self.is_train:
                self.positive_pairs[key] = self.positive_pairs[key][cutoff_idx:]
            else:
                self.positive_pairs[key] = self.positive_pairs[key][:cutoff_idx]
        
        keys = list(self.positive_pairs.keys())
        for key in keys:
            if key not in self.include_class:
                del self.positive_pairs[key]

        self.positive_pairs = [(key, *pair)  for key in self.positive_pairs for pair in self.positive_pairs[key]]
        self.negative_pairs = list(permutations(self.groups.keys(), 2))
        self.negative_pairs = [x for x in self.negative_pairs if x[0] in self.include_class]
        print('Total positive pairs:', len(self.positive_pairs))
        np.random.shuffle(self.positive_pairs)
        np.random.shuffle(self.negative_pairs)

class OEMAdapterDataset(OEMDataset):
    def __init__(
        self, 
        root:str, 
        class_idx: int,
        mean:Iterable[float]=[0.485, 0.456, 0.406], 
        std:Iterable[float]=[0.229, 0.224, 0.225], 
        resize: Tuple[int, int] = (448, 448),
        max_classes: int = 12, # including background
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.75,
        validation_ratio: float = 0.1,
        is_train: bool = True,
    ):
        self.class_idx = class_idx
        self.validation_ratio = validation_ratio
        
        super().__init__(root, mean, std, resize, max_classes, patch_size, mask_ratio, is_train)
        
        if not self.is_train:
            self.color_palette = np.array([[0, 0, 0], [255, 255, 255]])

    def _preload_dataset(self):
        self.images = []
        self.labels = []
        for img_path, label_path in tqdm(self.paths, desc='Caching images and grouping labels'):
            img = self._load_img(img_path)
            label = self._load_lbl(label_path)

            mask = label == self.class_idx
            c_pixels = mask.sum()
            if c_pixels > 0:
                self.images.append(img)
                self.labels.append(mask.astype(np.uint8))
        
        cut_off_idx = int(len(self.images) * (1 - self.validation_ratio))
        if self.is_train:
            self.images = self.images[:cut_off_idx]
            self.labels = self.labels[:cut_off_idx]
        else:
            self.images = self.images[cut_off_idx:]
            self.labels = self.labels[cut_off_idx:]

    def _filter_pairs(self):
        pass
    
    def _generate_pairs(self):
        pass

    def __len__(self):
        return len(self.images)

    def _generate_color_palette(self):
        return np.random.randint(0, 256, (2, 3)) # background and one class

    def _lbl_random_color(self, label: np.ndarray, color_palette: np.ndarray):
        result = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for i in range(2):
            result[label == i] = color_palette[i]
        return result

    def __getitem__(self, idx):
        img = self.images[idx]
        ori_label = self.labels[idx]

        if self.is_train:
            color_palette = self._generate_color_palette()
        else:
            color_palette = self.color_palette

        label = self._lbl_random_color(ori_label, color_palette)

        img, label, ori_label = self._augment([img], [label], [ori_label])
        img = self._to_img_tensor(img[0])
        label = self._to_img_tensor(label[0])
        ori_label = torch.FloatTensor(ori_label[0])
        
        mask = self._generate_mask((img.shape[1] * 2, img.shape[2]), True)
        valid = torch.ones(label.shape[0], label.shape[1] * 2, label.shape[2])
        seg_type = torch.zeros([1])
        color_palette = torch.FloatTensor(color_palette)
        return img, label, mask, valid, seg_type, ori_label, color_palette

    
if __name__ == '__main__':
    dataset = OEMAdapterDataset('/home/steve/Datasets/OpenEarthMap-FSS/valset', is_train=True, class_idx=9)
    # for i in tqdm(range(len(dataset))):
    #     a = dataset[i]
    # img, label = dataset[0]
    # img, label, mask, valid, seg_type, ori_label, color_palette = dataset[0]
    # for i in range(len(dataset))
    # img = img.cpu().numpy()
    # label = label.cpu().numpy()
    # img = (((img * dataset.std) + dataset.mean) * 255).astype(np.uint8)
    # lbl = (((label * dataset.std) + dataset.mean) * 255).astype(np.uint8)
    # print(img.shape, label.shape)
    # print(mask)
