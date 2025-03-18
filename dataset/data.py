from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random
# import ipdb
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import yaml
import argparse
 
class BSDataset(Dataset):
    def __init__(self, name, root, mode, size=None, nsample=None):
        self.name = name
        self.root = "/data/grassset2_80/"
        self.id_path = self.root + "train.txt"
        self.mode = mode
        # crop size
        self.size = size
        self.strong_aug = strong_img_aug()  # 初始化强增强
        self.val = self.root + "val/"
        val_path = self.root + "val.txt"
        if mode == 'train_l' or mode == 'train_u':
            with open(self.id_path, 'r') as f:
            # with open(mini_path, 'r') as f:
                self.ids = f.read().splitlines() 
            random.shuffle(self.ids)
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()


    def __getitem__(self, item):
        id = self.ids[item]
        if self.mode == 'train_l' or self.mode == 'train_u':
            img = Image.open(self.root + "train/image/" + id+'.png').convert('RGB')
            mask = Image.fromarray(np.array(Image.open(self.root + "train/mask/" + id+'.png' ))/255)
            # img, mask = hflip(img, mask, p=0.5)  # 随机水平翻转
            # img = self.strong_aug(img)  # 随机应用一些图像增强
            mask = torch.from_numpy(np.array(mask)).long()
            cutmix_box = obtain_cutmix_box(img.size[0], p=0.5)  
            img= normalize(img)
            return img, mask, cutmix_box

        else:
            img = Image.open( self.root + "val/image/" + id+'.png').convert('RGB')
            mask = Image.fromarray(np.array(Image.open(self.root + "val/mask/" + id+'.png' ))/255)
            img, mask = normalize(img, mask)
            return img, mask, id


    def __len__(self):
        return len(self.ids)