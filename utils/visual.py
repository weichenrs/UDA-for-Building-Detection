import time 
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

def plotfig(loss_hist, savepath):
    lt = time.localtime(time.time())
    yyyy = str(lt.tm_year)
    mm = str(lt.tm_mon)
    dd = str(lt.tm_mday)
    hh = str(lt.tm_hour)
    mn = str(lt.tm_min)
    sc = str(lt.tm_sec)
    timename = '-'+yyyy+'-'+mm+'-'+dd+'-'+hh+'-'+mn+'-'+sc
    # print(timename)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.plot(range(len(loss_hist)), loss_hist[:,0])
    ax2.plot(range(len(loss_hist)), loss_hist[:,1])
    ax1.set_title("Average loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("loss")
    ax2.set_title("Average IoU vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    plt.savefig(osp.join(savepath,'loss&miu'+timename+'.png'))

if __name__ == "__main__":
    npz = np.load('../Snap/Potsdam_loss.npz')
    loss_hist = npz['loss_hist']
    savepath = '../fig'
    plotfig(loss_hist,savepath)
