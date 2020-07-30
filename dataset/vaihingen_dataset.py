import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class VaihingenDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='train'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.id_to_trainid = {255: 1}
        self.set = set
        if not max_iters==None:
            n_repeat = int(max_iters / len(self.img_ids))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        if self.set == 'test':
            for name in self.img_ids:
                img_file = osp.join(self.root, "images/%s" % name)
                self.files.append({
                    "img": img_file,
                    "name": name
                })
        else:
            for name in self.img_ids:
                img_file = osp.join(self.root, "images/%s" % name)
                label_file = osp.join(self.root, "labels/%s" % name.replace('sat.jpg','mask.png'))
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        if not self.set == 'test':
            label = Image.open(datafiles["label"])
            label = label.resize(self.crop_size, Image.NEAREST)
            label = np.asarray(label, np.float32)
            label_copy = 255 * np.zeros(label.shape[0:2], dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label[:,:,0] == k] = v

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        image = image.resize(self.crop_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        if self.set == 'test':
            return image.copy(), name
        else:
            return image.copy(), label_copy.copy(), name
