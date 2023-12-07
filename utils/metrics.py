import os

import SimpleITK as sitk
import numpy as np
import torch
from medpy import metric


def calculate_metric_per_case(pred, gt, affine=None):
    """
    计算指标，返回
        DSC,
        Jaccard coefficient,
        95th percentile of the Hausdorff Distance
    :param pred: 预测图
    :param gt: 真实标签
    :param affine: 仿射矩阵
    :return: dice, jc, hd
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    if pred.sum() == 0:
        dice = 0
        jc = 0
        hd = 0
    else:
        dice = metric.binary.dc(pred, gt)  # DSC
        jc = metric.binary.jc(pred, gt)  # Jaccard coefficient
        hd = metric.binary.hd95(pred, gt, voxelspacing=abs(affine))  # 95th percentile of the Hausdorff Distance.

    return dice, jc, hd


def cal_dice(prediction, label):
    """
    计算 dice, 2 * (A ∩ B) / (A + B)
    :param prediction: 预测图
    :param label: 真实标签
    :return: dice
    """
    label = label.float()
    smooth = 1e-5
    intersect = torch.sum(prediction * label)
    y_sum = torch.sum(label * label)
    z_sum = torch.sum(prediction * prediction)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return dice


def cal_iou(prediction, label):
    """
    计算 iou, A ∩ B / (A + B - A ∩ B)
    :param prediction: 预测图
    :param label: 真实标签
    :return: iou
    """
    label = label.float()
    smooth = 1e-5
    intersect = torch.sum(prediction * label)
    y_sum = torch.sum(label * label)
    z_sum = torch.sum(prediction * prediction)
    iou = (intersect + smooth) / (z_sum + y_sum - intersect + smooth)
    return iou


class AverageMetric(object):
    def __init__(self):
        self.std = []
        self.mean = []
        self.metrics = []

    def update(self, m):
        self.metrics.append(m)

    def get_mean(self):
        self.mean = np.mean(self.metrics)

    def get_std(self):
        self.std = np.std(self.metrics)

    def get_all(self):
        self.get_std()
        self.get_mean()
        return self.mean, self.std, self.metrics


class AverageMetricMore(object):
    def __init__(self):
        self.std = {}
        self.mean = {}
        self.metrics = {}

    def update(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_mean(self):
        for metric_name, values in self.metrics.items():
            self.mean[metric_name] = np.mean(values)

    def get_std(self):
        for metric_name, values in self.metrics.items():
            self.std[metric_name] = np.std(values)

    def get_all(self):
        self.get_std()
        self.get_mean()
        return self.mean, self.std, self.metrics


class EvaluateOf3D(object):

    def __init__(self, infer_path=None, label_path=None):
        self.infer_path = infer_path
        self.label_path = label_path

    def calculate_hd(self, pred_masks, true_masks, mask):
        hd = metric.binary.hd(pred_masks, true_masks)
        a, b, c = mask.shape  # 获取图像深度、高度和宽度信息，mask只要是3维的图像即可，可以考虑加在数据读入的时候直接获取成一个数组，二维同理
        return hd / np.sqrt(a * a + b * b + c * c)

    def calculate_dice(self, pred_data, label_data):
        intersection = np.logical_and(pred_data, label_data)
        tp = np.sum(intersection)
        fp = np.sum(pred_data) - tp
        fn = np.sum(label_data) - tp

        dice = (2 * tp) / (2 * tp + fp + fn)

        return dice

    def calculate_miou(self, pred_masks, true_masks, num_classes=2):
        num_masks = len(pred_masks)
        intersection = np.zeros(num_classes)
        union = np.zeros(num_classes)

        for i in range(num_masks):
            pred_mask = pred_masks[i]
            true_mask = true_masks[i]

            for cls in range(num_classes):
                pred_cls = pred_mask == cls
                true_cls = true_mask == cls

                intersection[cls] += np.logical_and(pred_cls, true_cls).sum()
                union[cls] += np.logical_or(pred_cls, true_cls).sum()

        iou = intersection / union
        miou = np.mean(iou)

        return miou

    def read_nifti(self, path):
        itk_img = sitk.ReadImage(path)
        itk_arr = sitk.GetArrayFromImage(itk_img)

        return itk_arr

    def get_result(self):
        dice_avg = 0
        hd_avg = 0
        iou_avg = 0
        num = 0
        for file in os.listdir(os.path.join(self.label_path)):
            infer_path = os.path.join(self.infer_path, file)
            label_path = os.path.join(self.label_path, file)  # 可能需要针对数据集位置等信息修改，同2D
            pred = self.read_nifti(infer_path)
            label = self.read_nifti(label_path)
            pred_1 = (pred == 1)
            label_1 = (label == 1)
            # print(pred.min(),pred.max())
            # print(pred_1.sum(),label_1.sum())
            if pred_1.sum() > 0 and label_1.sum() > 0:
                asd = metric.binary.asd(pred == 1, label == 1)
                dice = self.calculate_dice(pred == 1, label == 1)
                hd = self.calculate_hd(pred_1 == 1, label_1 == 1, label)
                iou = self.calculate_miou(pred_1 == 1, label_1 == 1)
            dice_avg += dice
            hd_avg += hd
            iou_avg += iou
            num = num + 1

        dice_avg = dice_avg / num
        hd_avg = hd_avg / num
        iou_avg = iou_avg / num
        return dice_avg, hd_avg, iou_avg


# https://github.com/yefan222/MICCAI-2023-STS
class Evaluate3D(object):
    """
    评估指标
    """
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def calculate_hd(self, pred_masks, true_masks, mask):
        hd = metric.binary.hd(pred_masks, true_masks)
        a, b, c = mask.shape  # 获取图像深度、高度和宽度信息，mask只要是3维的图像即可，可以考虑加在数据读入的时候直接获取成一个数组，二维同理
        return hd / np.sqrt(a * a + b * b + c * c)

    def calculate_dice(self, pred_data, label_data):
        intersection = np.logical_and(pred_data, label_data)
        tp = np.sum(intersection)
        fp = np.sum(pred_data) - tp
        fn = np.sum(label_data) - tp

        dice = (2 * tp) / (2 * tp + fp + fn)

        return dice

    def calculate_miou(self, pred_masks, true_masks):
        num_masks = len(pred_masks)
        intersection = np.zeros(self.num_classes)
        union = np.zeros(self.num_classes)

        for i in range(num_masks):
            pred_mask = pred_masks[i]
            true_mask = true_masks[i]

            for cls in range(self.num_classes):
                pred_cls = pred_mask == cls
                true_cls = true_mask == cls

                intersection[cls] += np.logical_and(pred_cls, true_cls).sum()
                union[cls] += np.logical_or(pred_cls, true_cls).sum()

        iou = intersection / union
        miou = np.mean(iou)

        return miou

    def get_result(self, pred, label):
        dice = 0
        hd = 0
        iou = 0
        pred_1 = (pred == 1)
        label_1 = (label == 1)
        if pred_1.sum() > 0 and label_1.sum() > 0:
            dice = self.calculate_dice(pred == 1, label == 1)
            hd = self.calculate_hd(pred_1 == 1, label_1 == 1, label)
            iou = self.calculate_miou(pred_1 == 1, label_1 == 1)

        return dice, iou, hd
