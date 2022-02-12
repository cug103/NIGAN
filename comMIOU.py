"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
from PIL import Image
import os

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        # print(imgPredict.shape, imgLabel.shape)
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def read_img(img_name):
    img = Image.open(img_name)
    #将读取的图像变为numpy矩阵
    np_img = np.array(img) # (224, 224, 3)
    """
    将3维变为4维矩阵，可以用reshape去做变换，
    因为只有一张图片，可以加一个数组，然后让它封装在一个列表中，
    它就会把列表的这一维也加上去，列表的这一维就是1.
    """
    np_img = np.asarray([np_img], dtype=np.int32) # (1, 224, 224, 3)
    return np_img


def computeall(labledir, predir):
    Sum_mIou = 0
    Sum_mpa = 0
    Sum_pa = 0
    filename = os.listdir(predir)
    # print(filename)
    # print(os.listdir('/mnt/data/Liuzhuoyue/data/road_seg/submits/DinkNet101_01_03_08_025_18_09_zero_900/'))
    for fn in filename:
        imgPredict = read_img(predir + fn)
        imgLabel = read_img(labledir + fn)
        # print(imgPredict)
        # imgPredict = read_img('/mnt/data/Liuzhuoyue/data/road_seg/submits/pre_ttt/1420.png')
        # imgLabel = read_img('/mnt/data/Liuzhuoyue/data/road_seg/submits/lable_ttt_out/1420.png')
        # imgPredict = np.array([255, 0, 255, 255, 0, 0])  # 可直接换成预测图片
        # imgLabel = np.array([0, 255, 255, 255, 0, 0])  # 可直接换成标注图片
        metric = SegmentationMetric(256)  # 3表示有3个分类，有几个分类就填几
        metric.addBatch(imgPredict, imgLabel)
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        Sum_mIou += mIoU
        Sum_mpa += mpa
        Sum_pa += pa
        # print('pa is : %f' % pa)
        # print('cpa is :')  # 列表
        # print(cpa)
        # print('mpa is : %f' % mpa)
        # print('%s is mIoU is : %f  mpa is : %f  pa is : %f' % (fn, mIoU, mpa, pa))
    print('mIoU is : %f' % (Sum_mIou / len(filename)))
    print('allmpa is : %f' % (Sum_mpa / len(filename)))
    print('allpa is : %f' % (Sum_pa / len(filename)))
    return (Sum_mIou / len(filename))


if __name__ == '__main__':
    Sum_mIou = 0
    Sum_mpa = 0
    Sum_pa = 0
    filename = os.listdir('/mnt/data/Liuzhuoyue/data/road_seg/submits/DinkNet34_lzy_01_07_08_15_48_24_zero_2000_00002_out_2d/')
    # print(filename)
    # print(os.listdir('/mnt/data/Liuzhuoyue/data/road_seg/submits/DinkNet101_01_03_08_025_18_09_zero_900/'))
    for fn in filename:
        imgPredict = read_img('/mnt/data/Liuzhuoyue/data/road_seg/submits/DinkNet34_lzy_01_07_08_15_48_24_zero_2000_00002_out_2d/'+fn)
        imgLabel = read_img('/mnt/data/Liuzhuoyue/data/road_seg/test_out/'+fn)
        # print(imgPredict)
        # imgPredict = read_img('/mnt/data/Liuzhuoyue/data/road_seg/submits/pre_ttt/1420.png')
        # imgLabel = read_img('/mnt/data/Liuzhuoyue/data/road_seg/submits/lable_ttt_out/1420.png')
        # imgPredict = np.array([255, 0, 255, 255, 0, 0])  # 可直接换成预测图片
        # imgLabel = np.array([0, 255, 255, 255, 0, 0])  # 可直接换成标注图片
        metric = SegmentationMetric(256)  # 3表示有3个分类，有几个分类就填几
        metric.addBatch(imgPredict, imgLabel)
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        Sum_mIou += mIoU
        Sum_mpa += mpa
        Sum_pa += pa
        # print('pa is : %f' % pa)
        # print('cpa is :')  # 列表
        # print(cpa)
        # print('mpa is : %f' % mpa)
        print('%s is mIoU is : %f  mpa is : %f  pa is : %f' % (fn, mIoU,mpa,pa))
    print('mIoU is : %f' % (Sum_mIou/len(filename)))
    print('allmpa is : %f' % (Sum_mpa/len(filename)))
    print('allpa is : %f' % (Sum_pa / len(filename)))

