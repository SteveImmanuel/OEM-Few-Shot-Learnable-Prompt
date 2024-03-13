import json, os

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

def main(args):
  os.makedirs("embed", exist_ok=True)
  train_img_dir = os.path.join(args['train_dataset_dir'], "images")
  train_files = [os.path.join(train_img_dir, file) for file in os.listdir(train_img_dir)]
  store_embed_array(train_files, "embed/train_embed.npz")
  # verifying
  train_data = np.load("embed/train_embed.npz")
  print(len(train_data['names']), train_data['embeds'].shape)

  val_img_dir = os.path.join(args['val_dataset_dir'], "images")
  val_files = [os.path.join(val_img_dir, file) for file in os.listdir(val_img_dir)]
  store_embed_array(val_files, "embed/val_embed.npz")
  # verifying
  val_data = np.load("embed/val_embed.npz")
  print(len(val_data['names']), val_data['embeds'].shape)

  # testing similarity
  embeds = val_data['embeds']
  names = val_data['names']
  test_index = 0
  similarity = cosine_similarity(embeds[None,0,:], embeds[1:])
  closest_idx = np.argmax(similarity)
  print(f"image {names[test_index]} is similar with {names[closest_idx+1]}")
  

if __name__ == '__main__':
  args = json.load(open('configs/base.json', 'r'))
  main(args)
