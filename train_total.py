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
from dataset.potsdam_dataset import PotsdamDataSet
from dataset.vaihingen_dataset import VaihingenDataSet
from dataset.vaihingen_pseudo import VaihingenPseudo
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from pseudo_label import _colorize_mask
import random
import math
from torchvision import transforms
from utils.visual import plotfig
import matplotlib.pyplot as plt

IMG_MEAN = np.array((98.933625, 108.389025, 99.84372), dtype=np.float32) #src
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLab V3 Plus")

    #dataset
    parser.add_argument("--data_dir_src", type=str, default='../data/tx/src/',
                        help="source dataset path.")
    parser.add_argument("--data_dir_tgt", type=str, default='../data/tx/sh/',
                        help="target val dataset path.")
    parser.add_argument("--data_dir_pse", type=str, default='../pseudo_total/',
                        help="target val dataset path.")              
    parser.add_argument("--data_list_src", type=str, default='../data/src.txt',
                        help="source dataset list file.")

    parser.add_argument("--data_list_tgt_val", type=str, default='../data/shval.txt',
                        help="target val dataset list file.")
    parser.add_argument("--data_list_tgt_train", type=str, default='../data/shtrain.txt',
                        help="target val dataset list file.")
    parser.add_argument("--data_list_tgt_test", type=str, default='../data/shtest.txt',
                        help="target val dataset list file.")

    parser.add_argument("--ignore_label", type=int, default=255,
                        help="the index of the label ignored in the training.")
    parser.add_argument("--input_size", type=str, default='512,512',
                        help="width and height of input images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    #network
    parser.add_argument("--batch_size", type=int, default=3,
                        help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--learning_rate", type=float, default=0.005,
                        help="base learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum.")
    parser.add_argument("--num_epoch", type=int, default=10,
                        help="number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.00005,
                        help="regularisation parameter for L2-loss.")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="The threshold of the pseudo label.")
    #result
    parser.add_argument("--snapshot_root", type=str, default='../snap/',
                        help="where to save snapshots of the model.")
    parser.add_argument("--log_root", type=str, default='../log/',
                        help="where to save snapshots of the model.")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="noise.")

    return parser.parse_args()

def main():

    """Create the model and start the training."""
    args = get_arguments() #从命令行获取参数

    T = np.linspace(0,1,args.num_epoch)
    weight = []
    for t in T:
        exp = math.exp(-5*(1-t)**2)
        weight.append(exp)

    lt = time.localtime(time.time())
    yyyy = str(lt.tm_year)
    mm = str(lt.tm_mon)
    dd = str(lt.tm_mday)
    hh = str(lt.tm_hour)
    mn = str(lt.tm_min)
    sc = str(lt.tm_sec)
    timename = '-'+yyyy+'-'+mm+'-'+dd+'-'+hh+'-'+mn+'-'+sc
    exp_name = 'Src2SH_uni_lr'+str(args.learning_rate)+'_ep'+str(args.num_epoch)+'_'+str(args.input_size.split(',')[0]+timename)
    # print(exp_name)
    args.snapshot_dir = os.path.join(args.snapshot_root, exp_name)
    if os.path.exists(args.snapshot_dir)==False:
        os.makedirs(args.snapshot_dir)
    if os.path.exists(args.log_root)==False:
        os.makedirs(args.log_root)
    if os.path.exists(args.data_dir_pse)==False:
        os.makedirs(args.data_dir_pse)

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

    #加载target中的所有数据
    tgt_loader = data.DataLoader(
                    VaihingenDataSet(args.data_dir_tgt, args.data_list_tgt_train, max_iters=len(src_loader), 
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN, set='train'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    test_loader = data.DataLoader(
                    VaihingenDataSet(args.data_dir_tgt, args.data_list_tgt_test, max_iters=len(src_loader),
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN, set='test'),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    #加载target中的验证集val
    val_loader = data.DataLoader(
                    VaihingenDataSet(args.data_dir_tgt, args.data_list_tgt_val, 
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN, set='val'),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pse_generator = data.DataLoader(
                    VaihingenDataSet(args.data_dir_tgt, args.data_list_tgt_test,
                    crop_size=input_size,
                    scale=False, mirror=False, mean=IMG_MEAN, set='test'),
                    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_batches = len(src_loader)
    optimizer = optim.SGD(train_params, lr= args.learning_rate,momentum = 0.9, weight_decay = 5e-4, nesterov = False)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = 0.9)
    criterion = SegmentationLosses().build_loss(mode='ce')
    num_steps = args.num_epoch*num_batches
    loss_hist = np.zeros((num_steps,7))
    index_i = -1
    OA_hist = 0.2 #miou大于该值,则对模型进行存储
    con_loss = torch.nn.MSELoss()

    for epoch in range(args.num_epoch):
        #if epoch==6:
        #    return
        print('1-weight:{}\n'.format(1-weight[epoch]))
        f.write('1-weight:{}\n'.format(1-weight[epoch]))
        print('lr is {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        if not epoch == 0:
            print('start generating pseudo label')
            starttime = time.time()
            Pseudo_net = DeepLab_net
            Pseudo_net.eval()
            Pseudo_net.cuda()
            dir1 = os.path.join(args.data_dir_pse,'/pseudo_lab/',str(epoch))
            dir2 = os.path.join(args.data_dir_pse,'/pseudo_col/',str(epoch))
            if not os.path.exists(dir1 or dir2):
                os.makedirs(dir1)
                os.makedirs(dir2)
            for index, batch in enumerate(pse_generator):
                image, name = batch
                # print(index, name)
                output = Pseudo_net(image.cuda()).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                # top1 = np.max(output,axis = 2)
                # top2 = np.min(output,axis = 2)
                # inter = top1 - top2
                inter = output[:,:,1] - output[:,:,0]
                pseudolab = np.asarray(np.argmax(output, axis=2), dtype=np.uint8) + 1 
                pseudolab[inter< args.threshold] = 0 #伪标签阈值
                pseudolab_col = _colorize_mask(pseudolab)
                pseudolab = Image.fromarray(pseudolab)
                name = name[0].split('/')[-1]
                pseudolab.save('%s/%s' % (dir1, name))
                pseudolab_col.save('%s/%s_color.png' % (dir2, name.split('.jpg')[0]))
                if (index+1) % 100 == 0:
                    print('%d processd' % (index+1))
            print('finish generating pseudo label')
            pseudotime = time.time() - starttime
            print('pseudo cost time: %.2f' % pseudotime)

        #加载target中的所以伪标签数据        
        if epoch == 0:
            pse_loader = data.DataLoader(
                            VaihingenDataSet(args.data_dir_tgt, args.data_list_tgt_test, max_iters=len(src_loader),
                            crop_size=input_size,
                            scale=False, mirror=False, mean=IMG_MEAN, set='val'),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        else:
            pse_loader = data.DataLoader(
                            VaihingenPseudo(args.data_dir_tgt, args.data_dir_pse, args.data_list_tgt_test, max_iters=len(src_loader),
                            crop_size=input_size,
                            scale=False, mirror=False, mean=IMG_MEAN, epoch=epoch),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        for batch_index, (src_data, tgt_data, test_data, pse_data) in enumerate(zip(src_loader, tgt_loader, test_loader, pse_loader)):
            index_i += 1
            tem_time = time.time()
            DeepLab_net.train()

            ###############################################################
            ###################### train with source ######################
            images, src_label, name = src_data
            images = images.cuda() #images shape: 2,3,512,512
            src_label = src_label.cuda() #src_label shape:2,512,512
            src_output = DeepLab_net(images) #src out shape:2,6,512,512
            # Src Segmentation Loss
            optimizer.zero_grad() 
            src_loss_value = criterion(src_output, src_label)
            _, predict_labels = torch.max(src_output, 1) #_保存最大值, predict_labels保存最大值对应的索引
            lbl_pred = predict_labels.detach().cpu().numpy()
            lbl_true = src_label.detach().cpu().numpy()
            metrics_batch = []
            for lt, lp in zip(lbl_true, lbl_pred):
                _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
                metrics_batch.append(mean_iu)
            srcmiu = np.mean(metrics_batch, axis=0)

            ###############################################################
            ###################### train with target ######################
            images, tgt_label, name = tgt_data #return image.copy(), label_copy.copy(),np.array(size),name
            images = images.cuda() #images shape: 2,3,512,512
            tgt_label = tgt_label.cuda() #src_label shape:2,512,512
            tgt_output = DeepLab_net(images) #src out shape:2,6,512,512
            # Tgt Segmentation Loss
            tgt_loss_value = criterion(tgt_output, tgt_label)
            _, predict_labels = torch.max(tgt_output, 1) #_保存最大值, predict_labels保存最大值对应的索引
            lbl_pred = predict_labels.detach().cpu().numpy()
            lbl_true = tgt_label.detach().cpu().numpy()
            metrics_batch = []
            for lt, lp in zip(lbl_true, lbl_pred):
                _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=args.num_classes)
                metrics_batch.append(mean_iu)
            tgtmiu = np.mean(metrics_batch, axis=0)

            ###############################################################
            ############## train with target consistency loss##############
            images, name = test_data
            images = images.cuda()
            tgt_t_input = images + torch.randn(images.size()).cuda() * args.noise
            tgt_s_input = images + torch.randn(images.size()).cuda() * args.noise
            angles = [90,180,270]
            alpha = random.choice(angles)
            #s网络的输入图片input 顺时针旋转
            angle1 = -alpha*math.pi/180 #顺时针
            theta1 = torch.tensor([
                    [math.cos(angle1),math.sin(-angle1),0],
                    [math.sin(angle1),math.cos(angle1) ,0]
                    ], dtype=torch.float)
            grid1 = F.affine_grid(theta1.unsqueeze(0).expand(args.batch_size,2,3), tgt_s_input.size()).cuda()
            tgt_s_input = F.grid_sample(tgt_s_input, grid1)
            tgt_s_output = DeepLab_net(tgt_s_input)
            #s网络的输出图片output 逆时针旋转
            angle2 = alpha*math.pi/180 #逆时针
            theta2 = torch.tensor([
                    [math.cos(angle2),math.sin(-angle2),0],
                    [math.sin(angle2),math.cos(angle2) ,0]
                    ], dtype=torch.float)
            grid2 = F.affine_grid(theta2.unsqueeze(0).expand(args.batch_size,2,3), tgt_s_output.size()).cuda()
            tgt_s_output = F.grid_sample(tgt_s_output, grid2)
            tgt_t_output = DeepLab_net(tgt_t_input)
            tgt_t_predicts = F.softmax(tgt_t_output, dim=1).transpose(1, 2).transpose(2, 3)
            tgt_s_predicts = F.softmax(tgt_s_output, dim=1).transpose(1, 2).transpose(2, 3)
            con_loss_value = con_loss(tgt_s_predicts, tgt_t_predicts)
            con_loss_value = weight[epoch] * con_loss_value

            ###############################################################
            ############### train with target pseudo lables ###############

            images, pse_label, name = pse_data
            images = images.cuda()
            pse_label = pse_label.cuda()
            pse_output = DeepLab_net(images)
            # Pseudo label Segmentation Loss
            pse_loss_value = criterion(pse_output, pse_label) #src_label shape: torch.Size([1, 512, 512])
            pse_loss_value = weight[epoch] * pse_loss_value
            if epoch == 0:
                pse_loss_value = 0 * pse_loss_value

            # TOTAL LOSS #
            total_loss = src_loss_value + tgt_loss_value + con_loss_value + pse_loss_value
            total_loss.backward()
            loss_hist[index_i,0] = total_loss.item()
            loss_hist[index_i,1] = src_loss_value.item()
            loss_hist[index_i,2] = tgt_loss_value.item()
            loss_hist[index_i,3] = con_loss_value.item()
            loss_hist[index_i,4] = pse_loss_value.item()
            loss_hist[index_i,5] = srcmiu
            loss_hist[index_i,6] = tgtmiu

            optimizer.step()

            batch_time = time.time()-tem_time
            printfrq = 10
            if (batch_index+1) % printfrq == 0:
                print('epoch %d/%d:  %d/%d time: %.2f srcmiu = %.1f tgtmiu = %.1f src_loss = %.3f tgt_loss = %.3f con_loss = %.3f pse_loss = %.3f\n'%(epoch+1,args.num_epoch, batch_index+1,num_batches, batch_time*printfrq, np.mean(loss_hist[index_i+1-printfrq:index_i+1,5])*100, np.mean(loss_hist[index_i+1-printfrq:index_i+1,6])*100, np.mean(loss_hist[index_i+1-printfrq:index_i+1,1]), np.mean(loss_hist[index_i+1-printfrq:index_i+1,2]), np.mean(loss_hist[index_i+1-printfrq:index_i+1,3]), np.mean(loss_hist[index_i+1-printfrq:index_i+1,4]) ))
                f.write('epoch %d/%d:  %d/%d time: %.2f srcmiu = %.1f tgtmiu = %.1f src_loss = %.3f tgt_loss = %.3f con_loss = %.3f pse_loss = %.3f\n'%(epoch+1,args.num_epoch, batch_index+1,num_batches, batch_time*printfrq, np.mean(loss_hist[index_i+1-printfrq:index_i+1,5])*100, np.mean(loss_hist[index_i+1-printfrq:index_i+1,6])*100, np.mean(loss_hist[index_i+1-printfrq:index_i+1,1]), np.mean(loss_hist[index_i+1-printfrq:index_i+1,2]), np.mean(loss_hist[index_i+1-printfrq:index_i+1,3]), np.mean(loss_hist[index_i+1-printfrq:index_i+1,4]) ))
                f.flush()

            if (batch_index+1) % (num_batches/2) == 0:
                #test_mIoU(f,model, data_loader, epoch,input_size, print_per_batches=10)
                #f是打开log.txt
                OA_new = test_mIoU(f, DeepLab_net, val_loader, epoch+1, input_size, print_per_batches=10)

                # Saving the models
                if OA_new > OA_hist:
                    f.write('Save Model\n')
                    print('Save Model')
                    model_name = exp_name+'_epoch_'+'_'+repr(epoch+1)+'batch'+repr(batch_index+1)+'miu_'+repr(int(OA_new*1000))+'.pth'
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

