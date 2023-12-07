import os
import time

import numpy as np
from scipy import ndimage
from skimage import measure

from utils.metrics import calculate_metric_per_case, AverageMetric
from utils.roi_dect import single_case_mc, single_case_vnet
from utils.tools import read_nii_image_data, read_nii_label_data, rescale


def erosion(label, size=(5, 5)):
    return ndimage.grey_dilation(label, size=size)


def dilation(label, size=(5, 5)):
    return ndimage.grey_erosion(label, size=size)


def cut_roi(image):
    img_1 = measure.label(image, connectivity=2)
    img_unique, img_counts = np.unique(img_1, return_counts=True)
    img_dict = dict(zip(img_unique, img_counts))
    img_dict = sorted(img_dict.items(), key=lambda x: x[1], reverse=True)
    teeth_num = img_dict[1][0]
    img_1[img_1 != teeth_num] = 0
    img_1[img_1 == teeth_num] = 1
    where_1 = np.where(img_1 == 1)
    x_min, x_max = np.min(where_1[0]), np.max(where_1[0])
    y_min, y_max = np.min(where_1[1]), np.max(where_1[1])
    z_min, z_max = np.min(where_1[2]), np.max(where_1[2])
    coords = (x_min, x_max, y_min, y_max, z_min, z_max)
    return img_1, coords


def val_all_case(model, image_path, label_path,
                 num_classes=2, norm_type=3,
                 patch_size=(112, 112, 80),
                 stride_xy=18, stride_z=4,
                 which_model=1,
                 is_average=True,
                 my_logger=None):
    """
    对所有的图像进行预测
    :param model: 模型
    :param image_path: 图像路径
    :param label_path: 标签路径
    :param num_classes: 类别数
    :param norm_type: 归一化类型
    :param patch_size: 滑动窗口大小
    :param stride_xy: 滑动窗口步长
    :param stride_z: 滑动窗口步长
    :param which_model: 哪个模型，1：单输入单输出，2：单输入多输出
    :param is_average: 是否求平均
    :param my_logger: 日志
    :return: 对应的指标
    """
    image_list = os.listdir(image_path)
    my_logger.info(f'开始验证，验证数据量为：{len(image_list)}')
    dice_average = AverageMetric()
    iou_average = AverageMetric()
    hd_average = AverageMetric()
    time_average = AverageMetric()

    dice_average_aug = AverageMetric()
    iou_average_aug = AverageMetric()
    hd_average_aug = AverageMetric()
    time_average_aug = AverageMetric()

    for idx, image_name in enumerate(image_list):
        st = time.time()

        image_full_name = os.path.join(image_path, image_name)
        label_full_name = os.path.join(label_path, image_name)

        # 读取原图，并进行归一化处理
        t1 = time.time()
        image_data, image_affine, spacing, (o_w, o_h, o_d) = read_nii_image_data(image_full_name,
                                                                                 is_rescale=True,
                                                                                 norm_type=norm_type)
        # 读取标签，不进行 统一 spacing 处理
        label_data, label_affine = read_nii_label_data(label_full_name, is_rescale=False)
        t2 = time.time()
        my_logger.info(f'现在处理的是 第 {idx} ：{image_full_name}, 读取数据用时：{t2 - t1} s')

        # 使用滑动窗口法进行预测
        if which_model == 1:
            prediction, score_map = single_case_vnet(model, image_data,
                                                     stride_xy, stride_z, patch_size,
                                                     num_classes=num_classes)
        elif which_model == 2:
            prediction, score_map = single_case_mc(model, image_data,
                                                   stride_xy, stride_z, patch_size,
                                                   num_classes=num_classes, is_average=is_average)

        # 预测图插值，回归到原始 spacing
        prediction = rescale(prediction, o_w, o_h, o_d, 'nearest')

        t3 = time.time()
        # 计算指标
        metric = calculate_metric_per_case(prediction, label_data, affine=image_affine[0][0])

        dice_average.update(metric[0])
        iou_average.update(metric[1])
        hd_average.update(metric[2])
        time_average.update(t3 - t2)

        my_logger.info(f'推理用时：{t3 - t2} s')
        my_logger.info(f'第 {idx} 个数据：{image_name} 的指标为：{metric}')

        # 腐蚀膨胀，获得最大连通域
        prediction_erosion = erosion(dilation(prediction, size=(2, 2, 2)), size=(4, 4, 4))

        # 根据最大连通域，获得最大ROI的坐标和对应的预测图
        prediction_erosion, cut_coords = cut_roi(prediction_erosion)

        # 根据坐标重新计算预测图
        new_prediction = np.zeros_like(prediction_erosion)
        cx_min = cut_coords[0] - 50
        if cx_min < 0:
            cx_min = 0
        cx_max = cut_coords[1] + 30

        cy_min = cut_coords[2] - 10
        if cy_min < 0:
            cy_min = 0
        cy_max = cut_coords[3] + 90

        cz_min = cut_coords[4] - 10
        if cz_min < 0:
            cz_min = 0
        cz_max = cut_coords[5] + 10

        my_logger.info(f' {image_name}, cut_coords: {cx_min, cx_max, cy_min, cy_max, cz_min, cz_max}')
        new_prediction[cx_min:cx_max, cy_min:cy_max, cz_min:cz_max] = prediction[cx_min:cx_max, cy_min:cy_max,
                                                                      cz_min:cz_max]

        prediction_aug = erosion(dilation(prediction, size=(5, 5, 5)), size=(5, 5, 5))
        t4 = time.time()

        metric_aug = calculate_metric_per_case(prediction_aug, label_data, affine=image_affine[0][0])

        dice_average_aug.update(metric_aug[0])
        iou_average_aug.update(metric_aug[1])
        hd_average_aug.update(metric_aug[2])
        time_average_aug.update(t4 - t2)
        my_logger.info(f'推理用时：{t4 - t2} s')
        my_logger.info(f'第 {idx} 个数据：{image_name} 腐蚀膨胀后的指标为：{metric_aug}')

        my_logger.info(f'{image_name} 推理结束, 耗时：{time.time() - st} s')

    dice_average.get_all()
    iou_average.get_all()
    hd_average.get_all()
    time_average.get_all()

    dice_average_aug.get_all()
    iou_average_aug.get_all()
    hd_average_aug.get_all()
    time_average_aug.get_all()

    my_logger.info(f'平均指标为：dice: {dice_average.mean} ± {dice_average.std}, '
                   f'iou: {iou_average.mean} ± {iou_average.std}, '
                   f'hd: {hd_average.mean} ± {hd_average.std}, '
                   f'time: {time_average.mean} ± {time_average.std}')
    my_logger.info(f'Aug 后平均指标为：dice: {dice_average_aug.mean} ± {dice_average_aug.std}, '
                   f'iou: {iou_average_aug.mean} ± {iou_average_aug.std}, '
                   f'hd: {hd_average_aug.mean} ± {hd_average_aug.std}, '
                   f'time: {time_average_aug.mean} ± {time_average_aug.std}')
    dict_metric = {
        "mean": {
            "dice": dice_average.mean,
            "iou": iou_average.mean,
            "hd": hd_average.mean,
            "time": time_average.mean
        },
        "std": {
            "dice": dice_average.std,
            "iou": iou_average.std,
            "hd": hd_average.std,
            "time": time_average.std
        }
    }
    dict_metric_aug = {
        "mean": {
            "dice": dice_average_aug.mean,
            "iou": iou_average_aug.mean,
            "hd": hd_average_aug.mean,
            "time": time_average_aug.mean
        },
        "std": {
            "dice": dice_average_aug.std,
            "iou": iou_average_aug.std,
            "hd": hd_average_aug.std,
            "time": time_average_aug.std
        }
    }
    return dict_metric, dict_metric_aug
