import logging
import numpy as np
import torch
from PIL import Image

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def get_logger(name: str, rank: int):
    # adapted from https://discuss.pytorch.org/t/ddp-training-log-issue/125808
    class NoOp:
        def __getattr__(self, *args):
            def no_op(*args, **kwargs):
                """Accept every signature by doing non-operation."""
                pass

            return no_op

    if rank == 0:
        return logging.getLogger(name)
    return NoOp()

def cmap_to_lbl(cmap: torch.tensor, color_palette: torch.tensor):
    H, W, C = cmap.shape
    N, _ = color_palette.shape
    dist_mat = torch.cdist(cmap.reshape(1, H * W, C), color_palette.unsqueeze(0), 2)
    dist_mat = dist_mat.reshape(H, W, N)
    label = torch.argmin(dist_mat, 2)
    result = torch.zeros_like(cmap)
    for i in range(N):
        result[label == i] = color_palette[i]
    return result, label

def calculate_iou(pred: torch.tensor, gt: torch.tensor, color_palette: torch.tensor):
    N, _ = color_palette.shape
    x = 0
    for i in range(N):
        pred_total = (pred == i)
        gt_total = (gt == i)
        intersection = (pred_total & gt_total).sum()
        union = pred_total.sum() + gt_total.sum() - intersection
        x += pred_total.sum()
        print(i, intersection, union, intersection/union)
    print(x)

if __name__ == '__main__':
    color_palette = [
        [0,0,0],
        [17, 141,215],
        [127,173,123],
        [225, 227,155],
        [185,122,87],
        [228,200,182],
        [150,150,150]
    ]
    color_palette = torch.FloatTensor(color_palette)

    img = np.array(Image.open('temp3.png'))
    img = torch.FloatTensor(img)
    dcmap, label = cmap_to_lbl(img, color_palette)
    dcmap = dcmap.numpy().astype(np.uint8)
    label = label.numpy().astype(np.uint8)
    label = Image.fromarray(label)
    label.save('pred.png')

    img = np.array(Image.open('temp2.png'))
    img = torch.FloatTensor(img)
    dcmap, label = cmap_to_lbl(img, color_palette)
    dcmap = dcmap.numpy().astype(np.uint8)
    label = label.numpy().astype(np.uint8)
    label = Image.fromarray(label)
    label.save('gt.png') #make sure to save as PNG for lossless compression


    img1 = np.array(Image.open('pred.png'))
    img1 = torch.FloatTensor(img1)
    img2 = np.array(Image.open('gt.png'))
    img2 = torch.FloatTensor(img2)
    
    calculate_iou(img1, img2, color_palette)
    # dcmap, label = cmap_to_lbl(img, color_palette)
    # dcmap = dcmap.numpy().astype(np.uint8)
    # label = label.numpy().astype(np.uint8)
    # dcmap = Image.fromarray(dcmap)
    # # dcmap.save('dcmap.jpg')
    # label = Image.fromarray(label)
    # label.save('gt.jpg')