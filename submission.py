import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import argparse
import os
import json
import os.path as osp
import torch as T
import zipfile
import inference as BaseInference
import inference_adapter as NovelInference
from tqdm import tqdm
from Painter.SegGPT.SegGPT_inference.models_seggpt import seggpt_vit_large_patch16_input896x448
from model import AdapterSegGPT
from combine import combine

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, help='path to ckpt', required=True)
    parser.add_argument('--prompt-img-dir', type=str, help='path to prompt image directory', required=True)
    parser.add_argument('--prompt-label-dir', type=str, help='path to prompt colored label directory', required=True)
    parser.add_argument('--dataset-dir', type=str, help='path to input image dir to be tested', required=True)
    parser.add_argument('--top-k', type=int, help='top-k prompts to use', default=2)
    parser.add_argument('--device', type=str, help='cuda or cpu', default='cuda')
    parser.add_argument('--outdir', type=str, help='path to output', default='out')
    return parser.parse_args()

def zip_directory(directory, zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, directory))

if __name__ == '__main__':
    args = get_args_parser()

    assert osp.exists(osp.join(args.ckpt_path, 'base.pt'))
    assert osp.exists(osp.join(args.ckpt_path, '8_test.pt'))
    assert osp.exists(osp.join(args.ckpt_path, '9_test.pt'))
    assert osp.exists(osp.join(args.ckpt_path, '10_test.pt'))
    assert osp.exists(osp.join(args.ckpt_path, '11_test.pt'))


    # # Base Classes
    # base_model = seggpt_vit_large_patch16_input896x448()
    # ckpt = T.load(osp.join(args.ckpt_path, 'base.pt'), map_location='cpu')
    # base_model.load_state_dict(ckpt['model_state_dict'])
    # print('Base checkpoint loaded')

    # base_model = base_model.to(args.device)
    # base_model.eval()

    # print('Running inference for base classes')
    # mapping = json.load(open(osp.join('mappings', 'test', 'vit.json')))
    # for input_image in tqdm(mapping):
    #     input = os.path.join(args.dataset_dir, input_image)
    #     prompt = [os.path.join(args.prompt_img_dir, file) for file in mapping[input_image][:args.top_k]]
    #     prompt_target = [os.path.join(args.prompt_label_dir, file.replace('.tif', '.png')) for file in mapping[input_image][:args.top_k]]
    #     outdir = osp.join(args.outdir, f'base')
    #     BaseInference.inference_image_with_crop(base_model, args.device, input, prompt, prompt_target, outdir, split=2)
    #     tgt_path = os.path.join(outdir, 'color', input_image.replace('.tif', '.png'))
    #     lbl_path = os.path.join(outdir, 'label', input_image.replace('.tif', '.png'))
    #     BaseInference.inference_stitch(base_model, args.device, input, tgt_path, lbl_path, prompt, prompt_target, outdir, split=2, width=4)

    # # Novel Classes
    # split_config = {
    #     8: 8,
    #     9: 4,
    #     10: 2,
    #     11: 2,
    # }
    # for i in range(8, 12):
    #     novel_model = AdapterSegGPT(base_model)
    #     ckpt = T.load(osp.join(args.ckpt_path, f'{i}_test.pt'), map_location='cpu')
    #     novel_model.image_tensor.data = ckpt['model_state_dict']['image_tensor']
    #     novel_model.mask_tensor.data = ckpt['model_state_dict']['mask_tensor']
    #     print(f'Adapter for class {i} loaded')

    #     novel_model = novel_model.to(args.device)
    #     novel_model.eval()

    #     print(f'Running inference for novel class {i}')
    #     for file in tqdm(os.listdir(args.dataset_dir)):
    #         outdir = osp.join(args.outdir, f'novel_{i}')
    #         NovelInference.inference_image_with_crop(novel_model, args.device, os.path.join(args.dataset_dir, file), i, outdir=outdir, split=split_config[i])
    #         tgt_path = os.path.join(outdir, 'color', file.replace('.tif', '.png'))
    #         lbl_path = os.path.join(outdir, 'label', file.replace('.tif', '.png'))
    #         if split_config[i] == 2:
    #             NovelInference.inference_stitch(novel_model, 'cuda', os.path.join(args.dataset_dir, file), i, tgt_path, lbl_path, outdir=outdir, split=2, width=4)

    # # Combine Base and Novel
    # filter_config = json.load(open(osp.join('mappings', 'test', 'filtering.json')))
    # print('Combining base and novel classes')
    # combine(
    #     class_idx=9,
    #     source_folder=osp.join(args.outdir, 'base', 'stitch'),
    #     target_folder=osp.join(args.outdir, 'novel_9'),
    #     rgb_img_folder=args.dataset_dir,
    #     outdir=osp.join(args.outdir, 'combined_9'),
    #     exclusion_list=filter_config['9']['filter_list'],
    #     force_overlay=filter_config['9']['force_overlay'],
    #     bg_check=True,
    # )
    # combine(
    #     class_idx=11,
    #     source_folder=osp.join(args.outdir, 'combined_9'),
    #     target_folder=osp.join(args.outdir, 'novel_11', 'stitch'),
    #     rgb_img_folder=args.dataset_dir,
    #     outdir=osp.join(args.outdir, 'combined_9_11'),
    #     exclusion_list=filter_config['11']['filter_list'],
    #     force_overlay=filter_config['11']['force_overlay'],
    #     bg_check=True,
    # )
    # combine(
    #     class_idx=10,
    #     source_folder=osp.join(args.outdir, 'combined_9_11'),
    #     target_folder=osp.join(args.outdir, 'novel_10', 'stitch'),
    #     rgb_img_folder=args.dataset_dir,
    #     outdir=osp.join(args.outdir, 'combined_9_11_10'),
    #     exclusion_list=filter_config['10']['filter_list'],
    #     force_overlay=filter_config['10']['force_overlay'],
    #     bg_check=True,
    # )
    # combine(
    #     class_idx=8,
    #     source_folder=osp.join(args.outdir, 'combined_9_11_10'),
    #     target_folder=osp.join(args.outdir, 'novel_8'),
    #     rgb_img_folder=args.dataset_dir,
    #     outdir=osp.join(args.outdir, 'combined_9_11_10_8'),
    #     exclusion_list=filter_config['8']['filter_list'],
    #     force_overlay=filter_config['8']['force_overlay'],
    #     bg_check=False,
    # )

    zip_directory(osp.join(args.outdir, 'combined_9_11_10_8', 'label'), osp.join(args.outdir, 'submission.zip'))
    
