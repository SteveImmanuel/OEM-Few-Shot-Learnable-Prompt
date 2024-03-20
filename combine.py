import os
import os.path as osp
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('Combine base and novel class', add_help=False)
    parser.add_argument('--class-idx', type=int, help='idx of the novel class', required=True)
    parser.add_argument('--source-folder', type=str, help='path to the source result (base)', required=True)
    parser.add_argument('--target-folder', type=str, help='path to the target result (novel)', required=True)
    parser.add_argument('--outdir', type=str, help='path to output directory', default='./')
    parser.add_argument('--exclusion-list', nargs='+', help='list of files to be excluded', default=[])
    parser.add_argument('--inclusion-list', nargs='+', help='list of files to be included', default=[])
    return parser.parse_args()

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

if __name__ == '__main__':
    args = get_args_parser()
    # class_idx = 11
    # exclusion_list = ['lima_15.png', 'slaskie_28.png', 'tonga_64.png']
    # inclusion_list = ['svaneti_13.png', 'shanghai_65.png', 'lohur_7.png', 'lohur_11.png', 'lohur_29.png', 'muenster_60.png']
    # source_folder = 'class_combined/with9new'
    # target_folder = 'class_novel/11_phase2_stitch'
    # outdir = 'class_combined/with9and11phase2muenster'

    color_base = osp.join(args.source_folder, 'color')
    label_base = osp.join(args.source_folder, 'label')
    color_source = osp.join(args.target_folder, 'color')
    label_source = osp.join(args.target_folder, 'label')

    os.makedirs(osp.join(args.outdir, 'color'), exist_ok=True)
    os.makedirs(osp.join(args.outdir, 'label'), exist_ok=True)

    for filename in tqdm(os.listdir(color_base)):
        if not osp.exists(osp.join(color_source, filename)):
            print('warning: file not found', filename)
            continue

        color_base_img = np.array(Image.open(osp.join(color_base, filename)))
        color_source_img = np.array(Image.open(osp.join(color_source, filename)))
        label_base_img = np.array(Image.open(osp.join(label_base, filename)))
        label_source_img = np.array(Image.open(osp.join(label_source, filename)))

        assert color_base_img.shape == color_source_img.shape
        assert label_base_img.shape == label_source_img.shape

        mask = label_source_img == args.class_idx
        target_img = color_base_img.copy()
        target_label = label_base_img.copy() 
        
        is_skip = False
        if len(args.inclusion_list) > 0 and filename not in args.inclusion_list:
            print('skipping', filename)
            is_skip = True

        if len(args.exclusion_list) > 0 and filename in args.exclusion_list:
            print('skipping', filename)
            is_skip = True

        if not is_skip:
            target_img[mask] = color_map[args.class_idx]
            target_label[mask] = args.class_idx

        target_img = Image.fromarray(target_img)
        target_img.save(osp.join(args.outdir, 'color', filename))
        target_label = Image.fromarray(target_label)
        target_label.save(osp.join(args.outdir, 'label', filename))
    