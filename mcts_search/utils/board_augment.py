from collections import OrderedDict

import torch
from torchvision.transforms import functional as TF


AUGMENTATION_FUNCS = OrderedDict()
INVERSE_FUNCS = OrderedDict()

def aug_wrapper(func):
    return AUGMENTATION_FUNCS.setdefault(func.__name__, func)

def inv_wrapper(func):
    return INVERSE_FUNCS.setdefault(func.__name__[4:], func)


@aug_wrapper
def rotate0(x):
    return x


@inv_wrapper
def inv_rotate0(x):
    return x


@aug_wrapper
def rotate90(x):
    return TF.rotate(x, 90)


@inv_wrapper
def inv_rotate90(x):
    return TF.rotate(x, 270)


@aug_wrapper
def rotate180(x):
    return TF.rotate(x, 180)


@inv_wrapper
def inv_rotate180(x):
    return TF.rotate(x, 180)


@aug_wrapper
def rotate270(x):
    return TF.rotate(x, 270)


@inv_wrapper
def inv_rotate270(x):
    return TF.rotate(x, 90)


@aug_wrapper
def vertical_flip(x):
    return torch.flip(x, [-2])


@inv_wrapper
def inv_vertical_flip(x):
    return torch.flip(x, [-2])


@aug_wrapper
def horizontal_flip(x):
    return torch.flip(x, [-1])


@inv_wrapper
def inv_horizontal_flip(x):
    return torch.flip(x, [-1])


@aug_wrapper
def diagonal_flip(x):
    return x.transpose(-1, -2)


@inv_wrapper
def inv_diagonal_flip(x):
    return x.transpose(-1, -2)


@aug_wrapper
def off_diagonal_flip(x):
    return inv_rotate90(rotate90(x).transpose(-1, -2))


@inv_wrapper
def inv_off_diagonal_flip(x):
    return inv_rotate90(rotate90(x).transpose(-1, -2))
