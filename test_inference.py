import argparse
import os
import time

import numpy as np
import torch
from loguru import logger
from scipy import ndimage
from skimage import measure

from networks.net_factory import net_factory
from utils.roi_dect import single_case_vnet, single_case_mc
from utils.tools import make_dir, read_nii_image_data, rescale, save_nii_data


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_basic_path', type=str,
                        default=r'D:\LMQ\Experiments\STS2023-experiments',
                        help='训练文件夹基本路径')

    parser.add_argument("--test_image_path", type=str,
                        default=r'../Data/STS-Data/rematch/test',
                        help="val image path")
    parser.add_argument("--test_label_path", type=str,
                        default=r'../Data/STS-Data/rematch/test_test/label',
                        help="val image path")

    parser.add_argument('--patch_size', type=tuple, default=(112, 112, 80), help='patch size per sample')

    parser.add_argument('--num_class', type=int, default=2, help='class of you want to segment')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use')

    args_ = parser.parse_args()

    return args_


def main():
    args = get_args()
    exp_name_list = [
        'semi-mt- WMCNet - CE+Dice-42'
    ]
    net_name_list = [
        'WMCNet'  # WMC-Net*
    ]

    stride_xy_z_list = [(48, 48)]
    # stride_z_list = [4, 8]

    model_name_list = ['final_epoch.pth']

    xlsx_save_path = r'./test_infer_results'
    make_dir(xlsx_save_path)

    main_log_path = os.path.join(xlsx_save_path, 'log/inference_')
    logger.add(main_log_path + '{time}.txt', rotation='00:00')

    for m_name in model_name_list:
        for stride_xy_z in stride_xy_z_list:
            stride_xy, stride_z = stride_xy_z[0], stride_xy_z[1]
            logger.info(f'现在处理的是：{m_name}，stride_xy: {stride_xy}, stride_z: {stride_z}')
            start_time = time.time()
            save_infer_name = 'test-' + m_name.split('.')[0] + '-'
            save_infer_name += str(stride_xy) + '-' + str(stride_z) + '-'

            for idx, (exp_name, net_name) in enumerate(zip(exp_name_list, net_name_list)):
                logger.info(f'现在处理的是：{exp_name}，{net_name}')
                infer_one_model(save_exp_infer_name=save_infer_name,
                                exp_name=exp_name,
                                stride_xy=stride_xy,
                                stride_z=stride_z,
                                model_name=m_name,
                                net_name=net_name,
                                args=args)
            logger.info(f' 用时：{time.time() - start_time} s')


def infer_one_model(save_exp_infer_name, exp_name, stride_xy, stride_z, model_name, net_name, args=None):
    main_st_time = time.time()

    snapshot_path = "../Experiments/STS-inference/" + save_exp_infer_name + exp_name
    log_path = os.path.join(snapshot_path, 'log/inference_')
    logger.add(log_path + '{time}.txt', rotation='00:00')

    inference_save_path = os.path.join(snapshot_path, 'infers_common')
    inference_augmentation_save_path = os.path.join(snapshot_path, 'infers_augmentation')
    model_path = os.path.join(args.model_basic_path, exp_name, model_name)

    make_dir(snapshot_path)
    make_dir(inference_save_path)
    make_dir(inference_augmentation_save_path)

    logger.info(f'开始进行 推理，使用的模型为：{model_path}。')

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    model = create_model(net_name)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    which_model = 1 if net_name == 'vnet' or net_name == 'VNet_CBAM' else 2
    inference_all_case(model=model,
                       image_path=args.test_image_path,
                       num_classes=args.num_class,
                       patch_size=args.patch_size,
                       stride_xy=stride_xy,
                       stride_z=stride_z,
                       which_model=which_model,
                       save_result=True,
                       inference_save_path=inference_save_path,
                       inference_aug_save_path=inference_augmentation_save_path,
                       my_logger=logger)

    logger.info(f'总耗时：{time.time() - main_st_time} s')


def create_model(net_name):
    model = net_factory(net_type=net_name, in_chns=1, class_num=2, mode="test")
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


def inference_all_case(model, image_path,
                       num_classes=2, norm_type=3,
                       patch_size=(112, 112, 80),
                       stride_xy=18, stride_z=4,
                       which_model=1,
                       save_result=True,
                       inference_save_path=None,
                       inference_aug_save_path=None,
                       my_logger=None):
    """
    对所有的图像进行预测
    :param model: 模型
    :param image_path: 图像路径
    :param num_classes: 类别数
    :param norm_type: 归一化类型
    :param patch_size: 滑动窗口大小
    :param stride_xy: 滑动窗口步长
    :param stride_z: 滑动窗口步长
    :param which_model: 1: vnet, 2: mcnet
    :param save_result: 是否保存结果
    :param inference_save_path: 保存路径
    :param inference_aug_save_path: 数据增强保存路径
    :param my_logger: 日志
    :return: 对应图像列表的 affine
    """

    image_list = os.listdir(image_path)

    for idx, image_name in enumerate(image_list):
        st = time.time()
        name_idx = image_name.split('.')[0].split('t')[-1]
        image_full_name = os.path.join(image_path, image_name)

        # 读取原图，并进行归一化处理
        t1 = time.time()
        image_data, image_affine, spacing, (o_w, o_h, o_d) = read_nii_image_data(image_full_name,
                                                                                 is_rescale=True,
                                                                                 norm_type=norm_type)
        t2 = time.time()
        my_logger.info(f'现在处理的是 第 {idx} ：{image_full_name}, 读取数据用时：{t2 - t1} s, img_size:{o_w, o_h, o_d}')

        # 使用滑动窗口法进行预测
        if which_model == 1:
            prediction, score_map = single_case_vnet(model, image_data,
                                                     stride_xy, stride_z, patch_size,
                                                     num_classes=num_classes)

        elif which_model == 2:
            prediction, score_map = single_case_mc(model, image_data,
                                                   stride_xy, stride_z, patch_size,
                                                   num_classes=num_classes)

        # 预测图反归一化
        prediction = rescale(prediction, o_w, o_h, o_d, 'nearest')
        t3 = time.time()

        my_logger.info(f'推理用时：{t3 - t2} s')

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

        if cz_max - cz_min < 200 and name_idx in ['20', '40']:

            if cz_max > 300:
                cz_min -= 150
                if cz_min < 0:
                    cz_min = 0
            elif cz_min < 100:
                cz_max += 150
                if cz_max > 400:
                    cz_max = 400

        my_logger.info(f' {image_name}, cut_coords: {cx_min, cx_max, cy_min, cy_max, cz_min, cz_max}')

        new_prediction[cx_min:cx_max, cy_min:cy_max, cz_min:cz_max] = prediction[cx_min:cx_max, cy_min:cy_max,
                                                                      cz_min:cz_max]
        prediction_aug = erosion(dilation(new_prediction, size=(5, 5, 5)), size=(5, 5, 5))

        t4 = time.time()

        my_logger.info(f'推理用时：{t4 - t2} s')

        pred_name = os.path.join(inference_save_path, sub_name)
        pred_augmentation_name = os.path.join(inference_aug_save_path, sub_name)

        # 保存预测的标签
        if save_result:
            save_nii_data(prediction, image_affine, pred_name)
            save_nii_data(prediction_aug, image_affine, pred_augmentation_name)
            my_logger.info(f'预测标签保存到：{pred_name}')
        my_logger.info(f'{image_name} 推理结束, 耗时：{time.time() - st} s')

    return None


if __name__ == '__main__':
    main()
