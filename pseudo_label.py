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
import os, time
from PIL import Image
from utils.tools import *
from modeling.deeplab import *
import matplotlib.pyplot as plt
import torch.nn as nn

IMG_MEAN = np.array((98.933625, 108.389025, 99.84372), dtype=np.float32) #src
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
    parser.add_argument("--data_dir_tgt", type=str, default='../data/tx/sh/',
                        help="target dataset path.")
    parser.add_argument("--data_list_tgt_test", type=str, default='../data/shtest.txt',
                        help="target dataset list file.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="the index of the label to ignore in the training.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='../snap/Src2SH_lr0.001_ep10_1024/Src2SH_lr0.001_ep10_1024_epoch__10batch2081miu_702.pth',
                        help="restored model.")
    parser.add_argument("--snapshot_dir", type=str, default='../pseudo2/',
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

    pse_generator = data.DataLoader(
                    VaihingenDataSet(args.data_dir_tgt, args.data_list_tgt_test,
                    crop_size=(1024,1024),
                    scale=False, mirror=False, mean=IMG_MEAN, set='test'),
                    batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

    dir1 = os.path.join(args.snapshot_dir,'pseudo_lab')
    dir2 = os.path.join(args.snapshot_dir,'pseudo_col')
    if not os.path.exists(dir1 or dir2):
        os.makedirs(dir1)
        os.makedirs(dir2)
    print('start generating pseudo label')
    starttime = time.time()
    for index, batch in enumerate(pse_generator):

        image, name = batch
        output = model(image.cuda()).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        top = np.max(output,axis = 2)
        pseudolab = np.asarray(np.argmax(output, axis=2), dtype=np.uint8) + 1 
        pseudolab[top < args.threshold] = 0 #伪标签阈值
        pseudolab_col = _colorize_mask(pseudolab)
        output = Image.fromarray(pseudolab)
        name = name[0].split('/')[-1]
        output.save('%s/%s' % (dir1, name))
        pseudolab_col.save('%s/%s_color.png' % (dir2, name.split('.jpg')[0]))
        if (index+1) % 100 == 0:
            print('%d processd' % (index+1))
    print('finish generating pseudo label')
    pseudotime = time.time() - starttime
    print('pseudo cost time: %.2f' % pseudotime)
    #f.close()

if __name__ == '__main__':
    main()
