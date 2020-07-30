import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
import json

#指数滑动平均
class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)


def colorize_mask(mask):
    # mask: numpy array of the mask
    #各个类的标签的RGB值 对应json文件中的palette
    palette = [0,0,0, 255,255,255]
    #palette = [0, 0, 255, 255, 255, 0, 255, 0, 0, 255, 255, 255, 0, 255, 255, 0, 255, 0]

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

#语义分割损失
def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    #对于GTA5    label是8位1通道的      label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda()
    criterion = CrossEntropy2d().cuda()
    return criterion(pred, label)

#精确度计算
def _fast_hist(label_true, label_pred, n_class):
    # 标注计算类别范围以内的数，不在范围内的不考虑
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class=6):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = iu[1]
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def fast_hist(a, b, n):
    #a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为6）
    k = (a >= 0) & (a < n) #k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景）
    #np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    #分别为每个类别（在这里是6类）计算mIoU，hist的形状(n, n)
    #矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    #主要是因为CityScapes标签里面原类别太多，这样做把其他类别转换成算法需要的类别（共19类）和背景（标注为255）
    output = np.copy(input) #先复制一下输入图像
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1] #进行类别映射，最终得到的标签里面之后0-18这19个数加255（背景）
    return np.array(output, dtype=np.int64) #返回映射的标签

#计算Potsdam->Vaihingen的mIOU
def test_mIoU(f,model, data_loader, epoch,input_size, print_per_batches=10):

    model.eval()
    num_classes = 2
    num_batches = len(data_loader)
    hist = np.zeros((num_classes, num_classes)) #hist初始化为全零，在这里的hist的形状是[6,6]
    with open('./dataset/info.json','r') as fp: #读取info.json
      info = json.load(fp)
    name_classes = np.array(info['label'], dtype=np.str) #读取类别名称
    #interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    for ind, data in enumerate(data_loader):
        image, label = data[0].cuda(),data[1].squeeze()

        #_,outputs = model(image)
        outputs = model(image)
        #outputs = interp(outputs)
        _, predicted = torch.max(outputs, 1)

        pred = predicted.cpu().squeeze().numpy()
        label = label.numpy()

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:d}'.format(len(label.flatten()), len(pred.flatten()), ind))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % print_per_batches == 0:
            # print('{:d} / {:d}: {:0.2f}'.format(ind, num_batches, 100*np.mean(per_class_iu(hist))))
            # f.write('{:d} / {:d}: {:0.2f}\n'.format(ind, num_batches, 100*np.mean(per_class_iu(hist))))

            print('{:d} / {:d}: {:0.2f}'.format(ind, num_batches, 100*per_class_iu(hist)[1]  ))
            f.write('{:d} / {:d}: {:0.2f}\n'.format(ind, num_batches, 100*per_class_iu(hist)[1] ))

    mIoUs = per_class_iu(hist)
    # for ind_class in range(num_classes):
    #     f.write('\n===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    #     print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    f.write('\n epoch %d ===> mIoU: ' %(epoch) + str(round(mIoUs[1] * 100, 2))+'\n')
    f.flush()
    print('epoch %d ===> mIoU: ' %(epoch) + str(round(mIoUs[1] * 100, 2)))
    return mIoUs[1]

#计算SYNTHIA->CITYSCAPES的mIOU
def test_mIoU16(f,model, data_loader, epoch,input_size, print_per_batches=10):

    model.eval()
    num_classes = 16
    num_batches = len(data_loader)
    hist = np.zeros((num_classes, num_classes))
    with open('./dataset/info16.json','r') as fp:
      info = json.load(fp)
    name_classes = np.array(info['label'], dtype=np.str)
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    for ind, data in enumerate(data_loader):
        image, label = data[0].cuda(),data[1].squeeze()

        _,outputs = model(image)
        outputs = interp(outputs)
        _, predicted = torch.max(outputs, 1)

        pred = predicted.cpu().squeeze().numpy()
        label = label.numpy()

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:d}'.format(len(label.flatten()), len(pred.flatten()), ind))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % print_per_batches == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, num_batches, 100*np.mean(per_class_iu(hist))))
            f.write('{:d} / {:d}: {:0.2f}\n'.format(ind, num_batches, 100*np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        f.write('\n===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    f.write('\n epoch %d ===> mIoU: ' %(epoch) + str(round(np.nanmean(mIoUs) * 100, 2))+'\n')
    f.flush()
    print('epoch %d ===> mIoU: ' %(epoch) + str(round(np.nanmean(mIoUs) * 100, 2)))
    return np.nanmean(mIoUs)




class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight = None, ignore_index = self.ignore_label, size_average=self.size_average)
    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        #print(predict.size())
        #print(target.size())
        '''assert not target.requires_grad
        assert predict.dim() == 4   #torch.Size([1, 6, 512, 512])
        assert target.dim() == 3    #torch.Size([1, 512, 512])
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size() #n=1 c=6 h=512 w=512
        target_mask = (target >= 0) * (target != self.ignore_label) #torch.Size([1, 512, 512])
        target = target[target_mask]
        #print(target)
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss'''
        n, c, h, w = predict.size()         # n:batch_size, c:class
        predict = predict.view(-1, c)           # (n*h*w, c)
        target = target.view(-1)        # (n*h*w)
        # print('out', out.size(), 'target', target.size())
        loss = self.criterion(predict, target)
        return loss