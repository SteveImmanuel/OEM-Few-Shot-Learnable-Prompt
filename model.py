import sys
sys.path.append('Painter/SegGPT/SegGPT_inference')

import torch as T
import torch.nn.functional as F
from Painter.SegGPT.SegGPT_inference.models_seggpt import SegGPT, seggpt_vit_large_patch16_input896x448
from typing import Tuple

class AdapterSegGPT(T.nn.Module):
    def __init__(self, seggpt_model: SegGPT, image_size: Tuple[int, int] = (448, 448)):
        super().__init__()

        self.seggpt = seggpt_model
        for param in self.seggpt.parameters():
            param.requires_grad = False

        self.image_size = image_size
        self.image_tensor = T.nn.Parameter(T.zeros((1, 3, self.image_size[0], self.image_size[1]), dtype=T.float32), requires_grad=True)
        self.mask_tensor = T.nn.Parameter(T.zeros((1, 3, self.image_size[0], self.image_size[1]), dtype=T.float32), requires_grad=True)
        
        T.nn.init.normal_(self.image_tensor, mean=0.0, std=0.02)
        T.nn.init.normal_(self.mask_tensor, mean=0.0, std=0.02)


    def forward(self, imgs, tgts, bool_masked_pos=None, valid=None, seg_type=None, merge_between_batch=-1):
        # imgs, tgts N, 3, H, W
        B = imgs.shape[0]

        img_tensor = self.image_tensor.expand(B, -1, -1, -1)
        mask_tensor = self.mask_tensor.expand(B, -1, -1, -1)
        
        input_imgs = T.cat((imgs, img_tensor), dim=2)
        input_tgts = T.cat((tgts, mask_tensor), dim=2)


        return self.seggpt.forward(input_imgs, input_tgts, bool_masked_pos, valid, seg_type, merge_between_batch)

if __name__ == '__main__':
    from data import OEMAdapterDataset

    dataset = OEMAdapterDataset('/home/steve/Datasets/OpenEarthMap-FSS/valset', is_train=True, class_idx=9)
    a, b, c, d, e,f,g = dataset[0]
    a=a
    b=b
    c=c
    d=d
    e=e

    model = seggpt_vit_large_patch16_input896x448()
    initial_ckpt = T.load('tuning2.pt', map_location='cpu')
    model.load_state_dict(initial_ckpt['model_state_dict'], strict=False)

    adapter = AdapterSegGPT(model)
    result = adapter(a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0), d.unsqueeze(0), e.unsqueeze(0))