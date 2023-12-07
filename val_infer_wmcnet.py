import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from scipy import ndimage
from skimage import measure

from networks.net_factory import net_factory
from utils.metrics import AverageMetric
from utils.metrics import calculate_metric_per_case
from utils.roi_dect import single_coord_case_vnet, single_coord_case_mc
from utils.tools import make_dir, read_nii_image_data, read_nii_label_data, rescale, save_nii_data, print_args


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str,
                        default='exp_2023_11_29_3-semi- WMCNet_NoCBAM-kaiming - CE+Dice-1234',
                        help='训练文件夹名称')

    parser.add_argument('--net_name', type=str, default='WMCNet_NoCBAM',
                        help='网络，vnet, VNet_CBAM, mcnet3d_v1, mcnet3d_v2, WMCNet, WMCNet_NoCBAM')

    parser.add_argument('--save_exp_infer_name', type=str,
                        default=r'20231129-final-10-',
                        help='推理保存文件夹名称， 在训练文件夹名称加个前缀，')
    parser.add_argument('--model_basic_path', type=str,
                        default=r'D:\LMQ\Experiments\STS2023-experiments',
                        help='训练文件夹基本路径')
    parser.add_argument('--model_name', type=str,
                        default=r'final_epoch.pth',
                        help='用来推理的模型 best_dice.pth, final_epoch.pth')

    parser.add_argument("--val_image_path", type=str,
                        default=r'../Data/STS-Data/rematch/sts_val/image',
                        help="val image path")
    parser.add_argument("--val_label_path", type=str,
                        default=r'../Data/STS-Data/rematch/sts_val/label',
                        help="val label path")

    parser.add_argument('--patch_size', type=tuple, default=(112, 112, 80), help='patch size per sample')

    parser.add_argument('--num_class', type=int, default=2, help='class of you want to segment')
    parser.add_argument('--stride_xy', type=int, default=18, help='滑动窗口步长')
    parser.add_argument('--stride_z', type=int, default=4, help='滑动窗口步长')

    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use')

    args_ = parser.parse_args()

    return args_


def main():
    main_st_time = time.time()
    args = get_args()

    snapshot_path = "../Experiments/STS-inference/" + args.save_exp_infer_name + args.exp_name + '_' + str(
        args.stride_xy)
    log_path = os.path.join(snapshot_path, 'log/inference_')
    logger.add(log_path + '{time}.txt', rotation='00:00')

    inference_save_path = os.path.join(snapshot_path, 'infers_common')
    inference_augmentation_save_path = os.path.join(snapshot_path, 'infers_augmentation')
    model_path = os.path.join(args.model_basic_path, args.exp_name, args.model_name)

    make_dir(snapshot_path)
    make_dir(inference_save_path)
    make_dir(inference_augmentation_save_path)

    logger.info(f'开始进行 推理，使用的模型为：{model_path}。')
    logger.info(print_args(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # if args.deterministic:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)

    model = create_model(args=args)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    metric_none, metric_aug = inference_all_case(model=model,
                                                 image_path=args.val_image_path,
                                                 label_path=args.val_label_path,
                                                 num_classes=args.num_class,
                                                 patch_size=args.patch_size,
                                                 stride_xy=args.stride_xy,
                                                 stride_z=args.stride_z,
                                                 save_result=True,
                                                 inference_save_path=inference_save_path,
                                                 inference_aug_save_path=inference_augmentation_save_path,
                                                 my_logger=logger)

    logger.info(f'平均指标为：dice: {metric_none["mean"]["dice"]}, '
                f'iou: {metric_none["mean"]["iou"]}, '
                f'hd: {metric_none["mean"]["hd"]}, '
                f'time: {metric_none["mean"]["time"]}')
    logger.info(f'Aug 后平均指标为：dice: {metric_aug["mean"]["dice"]}, '
                f'iou: {metric_aug["mean"]["iou"]}, '
                f'hd: {metric_aug["mean"]["hd"]}, '
                f'time: {metric_aug["mean"]["time"]}')
    logger.info(f'总耗时：{time.time() - main_st_time} s')


def create_model(args):
    model = net_factory(net_type=args.net_name, in_chns=1, class_num=args.num_class, mode="test")
    return model


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


def inference_all_case(model, image_path, label_path,
                       num_classes=2, norm_type=3,
                       patch_size=(112, 112, 80),
                       stride_xy=18, stride_z=4,
                       save_result=True,
                       inference_save_path=None,
                       inference_aug_save_path=None,
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
    :param save_result: 是否保存结果
    :param inference_save_path: 保存路径
    :param inference_aug_save_path: 数据增强保存路径
    :param my_logger: 日志
    :return: 对应图像列表的 affine
    """
    affine_list = []

    image_list = os.listdir(image_path)
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
        coords = (110, 440, 0, 345, 10, 320)

        image_full_name = os.path.join(image_path, image_name)
        label_full_name = os.path.join(label_path, image_name)

        # 读取原图，并进行归一化处理
        t1 = time.time()
        image_data, image_affine, spacing, (o_w, o_h, o_d) = read_nii_image_data(image_full_name,
                                                                                 is_rescale=True,
                                                                                 norm_type=norm_type)
        label_data, label_affine = read_nii_label_data(label_full_name, is_rescale=False)
        t2 = time.time()
        my_logger.info(f'现在处理的是 第 {idx} ：{image_full_name}, 读取数据用时：{t2 - t1} s')

        # 使用滑动窗口法进行预测
        prediction, score_map = single_coord_case_mc(model, image_data, coords,
                                                     stride_xy, stride_z, patch_size,
                                                     is_average=True,
                                                     num_classes=num_classes)

        # 预测图反归一化
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

        # 保存名字的前缀
        sub_name = image_full_name.split('\\')[-1]

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
        prediction_aug = erosion(dilation(new_prediction, size=(5, 5, 5)), size=(5, 5, 5))

        t4 = time.time()

        metric_aug = calculate_metric_per_case(prediction_aug, label_data, affine=image_affine[0][0])

        dice_average_aug.update(metric_aug[0])
        iou_average_aug.update(metric_aug[1])
        hd_average_aug.update(metric_aug[2])
        time_average_aug.update(t4 - t2)
        my_logger.info(f'推理用时：{t4 - t2} s')
        my_logger.info(f'第 {idx} 个数据：{image_name} 腐蚀膨胀后的指标为：{metric_aug}')

        # 保存对应的 affine
        affine_list.append(image_affine[0][0])

        pred_name = os.path.join(inference_save_path, sub_name)
        pred_augmentation_name = os.path.join(inference_aug_save_path, sub_name)

        # 保存预测的标签
        if save_result:
            save_nii_data(prediction.astype(np.uint8), image_affine, pred_name)
            save_nii_data(prediction_aug.astype(np.uint8), image_affine, pred_augmentation_name)
            my_logger.info(f'预测标签保存到：{pred_name}')
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


if __name__ == '__main__':
    main()
