import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from dataset.vaihingen_dataset import VaihingenDataSet
from collections import OrderedDict
import os
from PIL import Image
from utils.tools import *
from modeling.deeplab import *
import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def _colorize_mask(mask):
    # mask: numpy array of the mask
    #各个类的标签的RGB值 对应json文件中的palette
    palette = [255,0,0, 0,0,0, 255,255,255]
    '''zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)'''

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab")
    parser.add_argument("--data_dir", type=str, default='./data/SegmentationData/vaihingen/',
                        help="target dataset path.")
    parser.add_argument("--data_list", type=str, default='./dataset/Vaihingen_all.txt',
                        help="target dataset list file.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="the index of the label to ignore in the training.")
    parser.add_argument("--num-classes", type=int, default=6,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='./Snap/Potsdam2Vaihingen_epoch7batch1500tgt_miu_462.pth',
                        help="restored model.")
    parser.add_argument("--snapshot_dir", type=str, default='./Snap/pseudolabel',
                        help="Path to save result.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="The threshold of the pseudo label.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    #f = open(args.snapshot_dir+'Evaluation.txt', 'w')

    model = DeepLab(num_classes=args.num_classes,backbone='resnet',output_stride=16,sync_bn=True,freeze_bn=False)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
    testloader = data.DataLoader(VaihingenDataSet(args.data_dir, args.data_list, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                    batch_size=1, shuffle=False, pin_memory=True)


    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, _,_, name = batch
        output = model(image.cuda()).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        top1 = np.max(output,axis = 2)
        top2 = np.sort(output,axis = 2)[:,:,4]
        inter = top1 - top2
        pseudolab = np.asarray(np.argmax(output, axis=2), dtype=np.uint8) + 1 
        
        pseudolab[inter< args.threshold] = 0 #伪标签阈值
        pseudolab_col = _colorize_mask(pseudolab)
        output = Image.fromarray(pseudolab)

        name = name[0].split('/')[-1]
        dir1 = args.snapshot_dir + '/pseudolab'
        dir2 = args.snapshot_dir + '/pseudolab_col/'
        if not os.path.exists(dir1 or dir2):
            os.makedirs(dir1)
            os.makedirs(dir2)
        output.save('%s/%s' % (dir1, name))
        pseudolab_col.save('%s/%s_color.png' % (dir2, name.split('.')[0]))

    #f.close()

if __name__ == '__main__':
    main()
