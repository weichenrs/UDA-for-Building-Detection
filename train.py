import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import os.path as osp
import time
from utils.tools import *
from utils.loss import SegmentationLosses
from utils.visual import plotfig
from dataset.potsdam_dataset import PotsdamDataSet
from dataset.vaihingen_dataset import VaihingenDataSet
from dataset.vaihingen_pseudo import VaihingenPseudo
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
import random
import math
from torchvision import transforms
import matplotlib.pyplot as plt

# IMG_MEAN = np.array((97.535715, 97.54362, 91.88925), dtype=np.float32) #sh
IMG_MEAN = np.array((98.933625, 108.389025, 99.84372), dtype=np.float32) #src
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLab V3 Plus")

    #dataset
    parser.add_argument("--data_dir_src", type=str, default='../data/tx/src/',
                        help="source dataset path.")
    parser.add_argument("--data_list_src", type=str, default='../data/src.txt',
                        help="source dataset list file.")
    parser.add_argument("--data_dir_tgt_val", type=str, default='../data/tx/sh/',
                        help="target val dataset path.")
    parser.add_argument("--data_list_tgt_val", type=str, default='../data/shval.txt',
                        help="target val dataset list file.")
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    parser.add_argument("--input_size", type=str, default='1024,1024',
                        help="width and height of input images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    #network
    parser.add_argument("--batch_size", type=int, default=2,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=0.005,
                        help="base learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_epoch", type=int, default=10,
                        help="number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.00005,
                        help="regularisation parameter for L2-loss.")
    #result
    parser.add_argument("--snapshot_root", type=str, default='../snap/',
                        help="where to save snapshots of the model.")
    parser.add_argument("--log_root", type=str, default='../log/',
                        help="where to save snapshots of the model.")
    return parser.parse_args()

def main():

    """Create the model and start the training."""
    args = get_arguments() #从命令行获取参数
    lt = time.localtime(time.time())
    yyyy = str(lt.tm_year)
    mm = str(lt.tm_mon)
    dd = str(lt.tm_mday)
    hh = str(lt.tm_hour)
    mn = str(lt.tm_min)
    sc = str(lt.tm_sec)
    timename = '-'+yyyy+'-'+mm+'-'+dd+'-'+hh+'-'+mn+'-'+sc
    exp_name = 'Src2SH_srconly_lr'+str(args.learning_rate)+'_ep'+str(args.num_epoch)+'_'+str(args.input_size.split(',')[0]+timename)
    # print(exp_name)
    args.snapshot_dir = os.path.join(args.snapshot_root, exp_name)
    if os.path.exists(args.snapshot_dir)==False:
        os.makedirs(args.snapshot_dir)
    if os.path.exists(args.log_root)==False:
        os.makedirs(args.log_root)
    f = open(args.log_root + exp_name + '_log.txt', 'w')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    # Create network
    DeepLab_net = DeepLab(num_classes=args.num_classes,backbone='resnet',output_stride=16,sync_bn=False,freeze_bn=True)
    
    train_params = [{'params': DeepLab_net.get_1x_lr_params(), 'lr': args.learning_rate},
                    {'params': DeepLab_net.get_10x_lr_params(), 'lr': args.learning_rate * 10}]
 
    DeepLab_net = DeepLab_net.cuda()

    #加载source的数据集
    src_loader = data.DataLoader(
                    PotsdamDataSet(args.data_dir_src, args.data_list_src,
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    #加载source中的验证集val
    val_loader = data.DataLoader(
                    PotsdamDataSet(args.data_dir_tgt_val, args.data_list_tgt_val,
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_batches = len(src_loader)
    optimizer = optim.SGD(train_params, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=False)
    criterion = SegmentationLosses().build_loss(mode='ce')
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = 0.9)
    num_steps = args.num_epoch*num_batches
    loss_hist = np.zeros((num_steps,2))
    index_i = -1
    OA_hist = 0.01 #miou大于该值,则对模型进行存储

    for epoch in range(args.num_epoch):
        #if epoch==6:
        #    return
        print('lr is {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        for batch_index, (src_data) in enumerate(src_loader):
            index_i += 1

            tem_time = time.time()
            DeepLab_net.train()

            images, src_label, im_name = src_data
            images = images.cuda() #images shape: 2,3,512,512
            src_label = src_label.cuda() #src_label shape:2,512,512
            src_output = DeepLab_net(images) #src out shape:2,6,512,512
            
            optimizer.zero_grad() 
            cls_loss_value = criterion(src_output, src_label)
            _, predict_labels = torch.max(src_output, 1) #_保存最大值, predict_labels保存最大值对应的索引
            lbl_pred = predict_labels.detach().cpu().numpy()
            lbl_true = src_label.detach().cpu().numpy()
            metrics_batch = []
            for lt, lp in zip(lbl_true, lbl_pred):
                _,_,mean_iu,_ = label_accuracy_score(lt, lp, args.num_classes)
                metrics_batch.append(mean_iu)
            miu = np.mean(metrics_batch, axis=0)
            
            cls_loss_value.backward()
            loss_hist[index_i,0] = cls_loss_value.item()
            loss_hist[index_i,1] = miu

            optimizer.step()
            batch_time = time.time()-tem_time
            printfrq = 10
            if (batch_index+1) % printfrq == 0:
                print('epoch %d/%d:  %d/%d, time: %.2f, miu = %.1f, cls_loss = %.3f \n'%(epoch+1,args.num_epoch, batch_index+1,num_batches, batch_time*printfrq, np.mean(loss_hist[index_i+1-printfrq:index_i+1,1])*100, np.mean(loss_hist[index_i+1-printfrq:index_i+1,0])))
                f.write('epoch %d/%d:  %d/%d, time: %.2f, miu = %.1f, cls_loss = %.3f \n'%(epoch+1,args.num_epoch, batch_index+1,num_batches, batch_time*printfrq, np.mean(loss_hist[index_i+1-printfrq:index_i+1,1])*100, np.mean(loss_hist[index_i+1-printfrq:index_i+1,0])))
                f.flush()

            testfrq = (num_batches/2)
            if (batch_index+1) % testfrq == 0:
                #test_mIoU(f,model, data_loader, epoch,input_size, print_per_batches=10)
                #f是打开log.txt
                OA_new = test_mIoU(f, DeepLab_net, val_loader, epoch+1, input_size, print_per_batches=10)

                # Saving the models
                if OA_new > OA_hist:
                    f.write('Save Model\n')
                    print('Save Model')
                    model_name = exp_name+'_epoch'+repr(epoch+1)+'_'+repr((batch_index+1)/testfrq)+'_miu_'+repr(int(OA_new*1000))+'.pth'
                    torch.save(DeepLab_net.state_dict(), os.path.join(
                        args.snapshot_dir, model_name))
                    OA_hist = OA_new
        scheduler.step()
    f.close()
    torch.save(DeepLab_net.state_dict(), os.path.join(
        args.snapshot_dir, exp_name + '_final.pth'))
    np.savez(args.snapshot_dir + exp_name + '_loss&miu_stat.npz',loss_hist=loss_hist)
    plotfig(loss_hist,args.snapshot_dir)

if __name__ == '__main__':
    main()

