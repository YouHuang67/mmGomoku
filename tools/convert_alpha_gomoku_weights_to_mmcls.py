# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch


def convert_vit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('1.'):
            new_k = f'backbone{k[1:]}'
        elif k.startswith('2.'):
            new_k = f'decode_head{k[1:]}'
        else:
            raise ValueError(f'key {k} not supported')
        new_ckpt[new_k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='src model path or url')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_vit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(dict(state_dict=weight), args.dst)


if __name__ == '__main__':
    main()
