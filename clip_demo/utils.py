from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode, Pad, ToPILImage, CenterCrop
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)



class Ostu(object):
    def __call__(self, img):
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2RGB)
        return th2

def image_transform(image_size=224):
    transform = Compose([
        Ostu(),
        ToPILImage(),
        # Pad([30,30], padding_mode='edge'),
        Pad([30,30], padding_mode='constant', fill=255),
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform
