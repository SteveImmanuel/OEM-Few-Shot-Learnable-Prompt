from collections import defaultdict
from itertools import combinations
import json, os, random

import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

import torch

processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

def extract_img_embed(image:Image) -> torch.Tensor: # 1 x 768
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    return image_features

def store_embed_array(files:str, dest:str) -> None:
    embeds = []
    names = []
    for file in tqdm(sorted(files)):
        img = Image.open(file)
        embed = extract_img_embed(img)
        embeds.append(embed[0].detach().numpy())
        names.append(os.path.basename(file))
    np.savez(dest, names=names, embeds=np.array(embeds))

def generate_pair_for_train_mix():
    class_mapping = np.load("cache/class_mapping.npy", allow_pickle=True).item()
    file_contain_class = defaultdict(set)
    for class_id in class_mapping:
        if class_id == 0: continue
        for file in class_mapping[class_id]:
            file_contain_class[file].add(class_id)
            
    class_rank = defaultdict(list)
    for file in file_contain_class:
        class_rank[len(file_contain_class[file])].append(file)
        
    good_files = []
    for rank in range(5,9):
        good_files += class_rank[rank]

    train_data = np.load("cache/train_embed.npz")
    embeds, names = train_data['embeds'], train_data['names']
    name2idx = {name:i for i, name in enumerate(names)}
    idx2name = {i:name for i, name in enumerate(names)}

    pairs = []
    for label_file in good_files:
        idx = name2idx[label_file]
        similarity = cosine_similarity(embeds[None,idx,:], embeds)
        idx_order = np.argsort(similarity)[0,::-1]
        for novel_classes in combinations(file_contain_class[label_file], 4):
            novel_classes = list(novel_classes)
            
            # looking for novel_class
            used_file = set([idx])
            novel_files = []
            for novel_class in novel_classes:
                for cur_idx in idx_order:
                    if cur_idx in used_file: continue
                    cur_idx_filename = idx2name[cur_idx]
                    if novel_class in file_contain_class[cur_idx_filename]:
                        novel_files.append(cur_idx_filename)
                        used_file.add(cur_idx)
                        break
                        
            # looking for query file
            base_classes = set([clas for clas in file_contain_class[label_file] if clas not in novel_classes])
            for cur_idx in idx_order:
                if cur_idx in used_file: continue
                complete = True
                cur_idx_filename = idx2name[cur_idx]
                for base_class in base_classes:
                    if base_class not in file_contain_class[cur_idx_filename]:
                        complete = False
                        break
                if complete:
                    base_file = cur_idx_filename
                    break
            pairs.append((label_file, file_contain_class[label_file], base_classes, base_file, novel_classes, novel_files))
    
    train_ratio = 0.7
    split = int(len(pairs)*train_ratio)
    random.seed(1)
    random.shuffle(pairs)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    np.save('cache/train_mapping.npy', np.array(train_pairs, dtype=object))
    np.save('cache/val_mapping.npy', np.array(val_pairs, dtype=object))
    print("Train pairs: ", len(train_pairs), ", Val pairs: ", len(val_pairs), sep="")

def main(args):
    os.makedirs("cache", exist_ok=True)

    label_file = defaultdict(set)
    train_labels_folder = os.path.join(args['train_dataset_dir'], "labels")
    for file in tqdm(os.listdir(train_labels_folder)):
        filepath = os.path.join(train_labels_folder, file)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        for c in img.flatten():
            label_file[c].add(file)
    np.save('cache/class_mapping.npy', np.array(dict(label_file)))
    # test load
    class_mapping = np.load("cache/class_mapping.npy", allow_pickle=True).item()
    print(f"There are {len(class_mapping)} classes in the train set")

    train_img_dir = os.path.join(args['train_dataset_dir'], "images")
    train_files = [os.path.join(train_img_dir, file) for file in os.listdir(train_img_dir)]
    store_embed_array(train_files, "cache/train_embed.npz")
    # verifying
    train_data = np.load("cache/train_embed.npz")
    print(len(train_data['names']), train_data['embeds'].shape)

    val_img_dir = os.path.join(args['val_dataset_dir'], "images")
    val_files = [os.path.join(val_img_dir, file) for file in os.listdir(val_img_dir)]
    store_embed_array(val_files, "cache/val_embed.npz")
    # verifying
    val_data = np.load("cache/val_embed.npz")
    print(len(val_data['names']), val_data['embeds'].shape)

    # testing similarity
    embeds = val_data['embeds']
    names = val_data['names']
    similarity = cosine_similarity(embeds[None,0,:], embeds[1:])
    closest_idx = np.argmax(similarity)
    print(f"Testing: image {names[0]} is similar with {names[closest_idx+1]}")

    # caching mix dataset pairing
    generate_pair_for_train_mix()

if __name__ == '__main__':
    args = json.load(open('configs/head.json', 'r'))
    main(args)
