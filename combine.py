import os
import os.path as osp
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from typing import List

def get_args_parser():
    parser = argparse.ArgumentParser('Combine base and novel class', add_help=False)
    parser.add_argument('--class-idx', type=int, help='idx of the novel class', required=True)
    parser.add_argument('--img-folder', type=str, help='path to the RGB image', default='/disk3/steve/dataset/OpenEarthMap-FSS/testset/images')
    parser.add_argument('--source-folder', type=str, help='path to the source result (base)', required=True)
    parser.add_argument('--source-mask-label-folder', type=str, help='path to the source label for masking', default=None)
    parser.add_argument('--target-folder', type=str, help='path to the target result (novel)', required=True)
    parser.add_argument('--outdir', type=str, help='path to output directory', default='./')
    parser.add_argument('--exclusion-list', nargs='+', help='list of files to be excluded', default=[])
    parser.add_argument('--inclusion-list', nargs='+', help='list of files to be included', default=[])
    parser.add_argument('--force-overlay', nargs='+', help='list of files to force overlay', default=[])
    parser.add_argument('--bg-check', action='store_true', help='only overlay if the source is background')
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

def combine(
    class_idx: int,
    source_folder: str, 
    target_folder: str, 
    rgb_img_folder: str, 
    outdir: str, 
    inclusion_list: List = [],
    exclusion_list: List = [],
    bg_check: bool = False,
    force_overlay: List = [],
    source_mask_label_folder: str = None
):
    color_source = osp.join(source_folder, 'color')
    label_source = osp.join(source_folder, 'label')
    color_target = osp.join(target_folder, 'color')
    label_target = osp.join(target_folder, 'label')

    os.makedirs(osp.join(outdir, 'color'), exist_ok=True)
    os.makedirs(osp.join(outdir, 'label'), exist_ok=True)

    for filename in tqdm(os.listdir(color_source)):
        rgb_img = np.array(Image.open(osp.join(rgb_img_folder, filename.replace('png', 'tif'))).resize((1024, 1024)))
        color_source_img = np.array(Image.open(osp.join(color_source, filename)))
        label_source_img = np.array(Image.open(osp.join(label_source, filename)))
        
        is_skip = False
        if len(inclusion_list) > 0 and filename not in inclusion_list:
            print('skipping', filename)
            is_skip = True

        if len(exclusion_list) > 0 and filename in exclusion_list:
            print('skipping', filename)
            is_skip = True

        if not osp.exists(osp.join(color_target, filename)):
            print('warning: file not found', filename)
            is_skip = True

        target_img = color_source_img.copy()
        target_label = label_source_img.copy() 

        if not is_skip:
            color_target_img = np.array(Image.open(osp.join(color_target, filename)))
            label_target_img = np.array(Image.open(osp.join(label_target, filename)))

            assert color_source_img.shape == color_target_img.shape
            assert label_source_img.shape == label_target_img.shape

            mask = label_target_img == class_idx
            rgb_bg_mask = np.all(rgb_img == 0, axis=-1)
            mask = mask & ~rgb_bg_mask
            if bg_check and filename not in force_overlay:
                if source_mask_label_folder is None:
                    mask = mask & (label_source_img == 0)
                else:
                    label_mask = np.array(Image.open(osp.join(source_mask_label_folder, filename)))
                    mask = mask & (label_mask == 0)
            target_img[mask] = color_map[class_idx]
            target_label[mask] = class_idx

        target_img = Image.fromarray(target_img)
        target_img.save(osp.join(outdir, 'color', filename))
        target_label = Image.fromarray(target_label)
        target_label.save(osp.join(outdir, 'label', filename))

if __name__ == '__main__':
    args = get_args_parser()
    # class_idx = 11
    # exclusion_list = ['lima_15.png', 'slaskie_28.png', 'tonga_64.png']
    # inclusion_list = ['svaneti_13.png', 'shanghai_65.png', 'lohur_7.png', 'lohur_11.png', 'lohur_29.png', 'muenster_60.png']
    # source_folder = 'class_combined/with9new'
    # target_folder = 'class_novel/11_phase2_stitch'
    # outdir = 'class_combined/with9and11phase2muenster'
    combine(
        class_idx=args.class_idx,
        source_folder=args.source_folder,
        target_folder=args.target_folder,
        rgb_img_folder=args.img_folder,
        outdir=args.outdir,
        inclusion_list=args.inclusion_list,
        exclusion_list=args.exclusion_list,
        bg_check=args.bg_check,
        force_overlay=args.force_overlay,
        source_mask_label_folder=args.source_mask_label_folder
    )
    # color_source = osp.join(args.source_folder, 'color')
    # label_source = osp.join(args.source_folder, 'label')
    # color_target = osp.join(args.target_folder, 'color')
    # label_target = osp.join(args.target_folder, 'label')

    # os.makedirs(osp.join(args.outdir, 'color'), exist_ok=True)
    # os.makedirs(osp.join(args.outdir, 'label'), exist_ok=True)

    # for filename in tqdm(os.listdir(color_source)):
    #     rgb_img = np.array(Image.open(osp.join(args.img_folder, filename.replace('png', 'tif'))).resize((1024, 1024)))
    #     color_source_img = np.array(Image.open(osp.join(color_source, filename)))
    #     label_source_img = np.array(Image.open(osp.join(label_source, filename)))
        
    #     is_skip = False
    #     if len(args.inclusion_list) > 0 and filename not in args.inclusion_list:
    #         print('skipping', filename)
    #         is_skip = True

    #     if len(args.exclusion_list) > 0 and filename in args.exclusion_list:
    #         print('skipping', filename)
    #         is_skip = True

    #     if not osp.exists(osp.join(color_target, filename)):
    #         print('warning: file not found', filename)
    #         is_skip = True

    #     target_img = color_source_img.copy()
    #     target_label = label_source_img.copy() 

    #     if not is_skip:
    #         color_target_img = np.array(Image.open(osp.join(color_target, filename)))
    #         label_target_img = np.array(Image.open(osp.join(label_target, filename)))

    #         assert color_source_img.shape == color_target_img.shape
    #         assert label_source_img.shape == label_target_img.shape

    #         mask = label_target_img == args.class_idx
    #         rgb_bg_mask = np.all(rgb_img == 0, axis=-1)
    #         mask = mask & ~rgb_bg_mask
    #         if args.bg_check and filename not in args.force_overlay:
    #             if args.source_mask_label_folder is None:
    #                 mask = mask & (label_source_img == 0)
    #             else:
    #                 label_mask = np.array(Image.open(osp.join(args.source_mask_label_folder, filename)))
    #                 mask = mask & (label_mask == 0)
    #         target_img[mask] = color_map[args.class_idx]
    #         target_label[mask] = args.class_idx

    #     target_img = Image.fromarray(target_img)
    #     target_img.save(osp.join(args.outdir, 'color', filename))
    #     target_label = Image.fromarray(target_label)
    #     target_label.save(osp.join(args.outdir, 'label', filename))
    