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
#from model.SEAN import SEANet
from dataset.vaihingen_dataset import VaihingenDataSet
from collections import OrderedDict
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
from utils.tools import *
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
import matplotlib.pyplot as plt
import torch.nn as nn

IMG_MEAN = np.array((97.535715, 97.54362, 91.88925), dtype=np.float32)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab")
    parser.add_argument("--data_dir", type=str, default='../data/tx/sh/',
                        help="target dataset path.")
    parser.add_argument("--data_list", type=str, default='../data/shtest.txt',
                        help="target dataset list file.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="the index of the label to ignore in the training.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--restore-from", type=str, default='../snap/Src2SH_lr0.002_ep10_1024/Src2SH_lr0.002_ep10_1024_epoch__9batch4162miu_734.pth',
                        help="restored model.")
    parser.add_argument("--snapshot_dir", type=str, default='../map/Src2SH_lr0.002_ep10_1024/',
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    f = open(args.snapshot_dir+'Evaluation.txt', 'w')

    model = DeepLab(num_classes=args.num_classes,backbone='resnet',output_stride=16,sync_bn=True,freeze_bn=False)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()
    testloader = data.DataLoader(VaihingenDataSet(args.data_dir, args.data_list, crop_size=(1024, 1024), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                    batch_size=1, shuffle=False, pin_memory=True)

    input_size_target = (1024,1024)
    interp = nn.Upsample(size=(1024,1024), mode='bilinear')

    # test_mIoU(f,model, testloader, 0,input_size_target,print_per_batches=10)

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, _, name = batch
        output = model(image.cuda()).cpu().data[0].numpy() #([1, 2, 1024, 1024])

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        dir1 = args.snapshot_dir + '/predict_lab'
        dir2 = args.snapshot_dir + '/predict_col/'
        if not os.path.exists(dir1 or dir2):
            os.makedirs(dir1)
            os.makedirs(dir2)
        output.save('%s/%s' % (dir1, name))
        output_col.save('%s/%s_color.png' % (dir2, name.split('.png')[0]))

    f.close()

if __name__ == '__main__':
    main()
