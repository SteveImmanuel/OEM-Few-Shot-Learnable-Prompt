import os
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm

color_map = np.array([
    (0, 0,  0),
    (40, 130,  72),
    (255, 237, 2),
    (222, 173,  100),
    (215,  22, 194),
    (255, 255, 255),
    (59,  17, 243),
    (114,   6,  39),
    (218, 65, 32),
    (65, 105, 125),
    (0, 255, 127),
    (107, 142, 35), 
])
class_idx = 9
exclusion_list = ['lima_15.png', 'slaskie_28.png', 'tonga_64.png']
base_folder = 'class_base/vitmapfiltered_mim2_split2_new_stitch'
source_folder = 'class_novel/9_curbest'
outdir = 'class_combined/with9'

color_base = osp.join(base_folder, 'color')
label_base = osp.join(base_folder, 'label')
color_source = osp.join(source_folder, 'color')
label_source = osp.join(source_folder, 'label')

os.makedirs(osp.join(outdir, 'color'), exist_ok=True)
os.makedirs(osp.join(outdir, 'label'), exist_ok=True)

for filename in tqdm(os.listdir(color_base)):
    if not osp.exists(osp.join(color_source, filename)):
        continue

    color_base_img = np.array(Image.open(osp.join(color_base, filename)))
    color_source_img = np.array(Image.open(osp.join(color_source, filename)))
    label_base_img = np.array(Image.open(osp.join(label_base, filename)))
    label_source_img = np.array(Image.open(osp.join(label_source, filename)))

    assert color_base_img.shape == color_source_img.shape
    assert label_base_img.shape == label_source_img.shape

    mask = label_source_img == class_idx
    target_img = color_base_img.copy()
    target_label = label_base_img.copy() 
    
    if filename in exclusion_list:
        print('skipping', filename)
    else:
        target_img[mask] = color_map[class_idx]
        target_label[mask] = class_idx

    target_img = Image.fromarray(target_img)
    target_img.save(osp.join(outdir, 'color', filename))
    target_label = Image.fromarray(target_label)
    target_label.save(osp.join(outdir, 'label', filename))
    