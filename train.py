import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool,DinkNet34_lzy,DinkNet101_lzy
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
import computeIOU
from test import test_all
from torchsummary import summary



SHAPE = (256, 256)
ROOT = '/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/train_new/'
imagelist = [x for x in os.listdir(ROOT) if x.find('sat') != -1]
trainlist = [x[:-8] for x in imagelist]
NAME = 'DinkNet34_lzy_02_24_08_15_48_24_zero_2000_00002_out_4d'
BATCHSIZE_PER_CARD = 4
torch.cuda.set_device(1)
solver = MyFrame(DinkNet34_lzy, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
batchsize = 8

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=32)

miou = computeIOU.Evaluator(2)
mylog = open('/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/log/'+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = 2000
train_epoch_best_loss = 100.
best_miou = 0
accVal = 0
miouVal = 0
# D = DinkNet34_lzy()
# D = D.cuda()
# #
# summary(D, (3,256,256))

for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    IoU = 0.0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time()-tic), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time()-tic))
    print('train_loss:', train_epoch_loss)
    print('SHAPE:', SHAPE)
    print('nowbestmiou:', best_miou)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/weights/'+NAME+'.th')
        now_miou = test_all(NAME)
        if now_miou > best_miou:
            print('最大的MIOU从{}更新到{}'.format(best_miou, now_miou))
            best_miou = now_miou
            solver.save('/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/weights/' + NAME+ '_bestmiou.th')

    if no_optim > 48:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 24:
        if solver.old_lr < 5e-7:
            break
        solver.load('/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/weights/'+NAME+'.th')
        solver.update_lr(1.5, factor = True, mylog = mylog)
        no_optim = 0
    mylog.flush()
    
print('Finish!', file=mylog)
print('Finish!')
print('bestloss', train_epoch_best_loss)
mylog.close()