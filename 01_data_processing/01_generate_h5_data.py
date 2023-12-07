import os
import time

import h5py
import numpy as np
from loguru import logger

from utils.tools import make_dir, read_nii_image_data, read_data

log_path = './log/01_generate_h5_data'
logger.add(log_path + '{time}.txt', rotation='00:00')


def random_crop(image, label, output_size, crop_times=5, is_label=False):
    """
    Random crop image and label.
    :param image: image data
    :param label: label data
    :param output_size: the size of output image and label
    :param crop_times: the number of crop
    :param is_label: whether the data is label
    :return: cropped image and label
    """

    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape
    image_list, label_list = [], []

    if is_label:
        where_label = np.where(label > 0)
        x_max, x_min = np.max(where_label[0]), np.min(where_label[0])
        y_max, y_min = np.max(where_label[1]), np.min(where_label[1])
        z_max, z_min = np.max(where_label[2]), np.min(where_label[2])

        w1_min = int(x_min / 2)
        h1_min = int(y_min / 2)
        d1_min = int(z_min / 2)

        w1_max = x_max - output_size[0]
        h1_max = y_max - output_size[1]
        d1_max = z_max - output_size[2]

        if x_max + output_size[0] > w:
            w1_max = w - output_size[0]
        if y_max + output_size[1] > h:
            h1_max = h - output_size[1]
        if z_max + output_size[2] > d:
            d1_max = d - output_size[2]

        if w1_max < w1_min:
            w1_min, w1_max = w1_max, w1_min
        if h1_max < h1_min:
            h1_min, h1_max = h1_max, h1_min
        if d1_max < d1_min:
            d1_min, d1_max = d1_max, d1_min

        for i in range(crop_times):
            w1 = np.random.randint(w1_min, w1_max)
            h1 = np.random.randint(h1_min, h1_max)
            d1 = np.random.randint(d1_min, d1_max)
            logger.info(f'print the random coord: {w1, h1, d1}')

            label_list.append(label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
            image_list.append(image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
    else:
        # 观察数据
        x_min, x_max, y_min, y_max, z_min, z_max = 122, 437, 2, 343, 13, 315

        logger.info(f' x_min:{x_min}, x_max:{x_max}')
        logger.info(f' y_min:{y_min}, y_max:{y_max}')
        logger.info(f' z_min:{z_min}, z_max:{z_max}')

        x_delta = x_max - x_min
        y_delta = y_max - y_min
        z_delta = z_max - z_min

        lower_x = 0
        lower_y = 0
        lower_z = 0

        upper_x = w - output_size[0]
        upper_y = h - output_size[1]
        upper_z = d - output_size[2]

        if w > x_delta and h > y_delta and d > z_delta:
            lower_x = x_min
            lower_y = y_min
            lower_z = z_min

            upper_x = x_max - output_size[0]
            upper_y = y_max - output_size[1]
            upper_z = z_max - output_size[2]

        for i in range(crop_times):
            w1 = np.random.randint(lower_x, upper_x)
            h1 = np.random.randint(lower_y, upper_y)
            d1 = np.random.randint(lower_z, upper_z)
            logger.info(f'print the random coord: {w1, h1, d1}')

            label_list.append(label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])
            image_list.append(image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]])

    return image_list, label_list


def covert_h5(images, labels, output_size, fn, crop_times=5, is_label=False):
    """
    Convert image and label to h5 file.
    :param images: image data
    :param labels: label data
    :param output_size: the size of output image and label
    :param fn: the name of h5 file
    :param crop_times: the number of crop
    :param is_label: whether the data is label
    :return: None
    """
    image_list, label_list = random_crop(images, labels, output_size, crop_times=crop_times, is_label=is_label)
    for file_i in range(len(image_list)):
        f = h5py.File(fn + str(file_i).zfill(3) + '_roi.h5', 'w')
        f.create_dataset('image', data=image_list[file_i])
        f.create_dataset('label', data=label_list[file_i])
        f.close()


def sts_label_data(data_list_path, h5_save_path, crop_times=6, output_size=(112, 112, 80), norm_type=1):
    """
    处理二分类数据
    """
    st_time = time.time()
    make_dir(h5_save_path)

    logger.info(f'开始处理 STS Labelled 二分类数据，从.nii.gz -> .h5')
    logger.info(f'列表路径为：{data_list_path}')
    logger.info(f'保存路径为：{h5_save_path}')
    output_size_ = output_size
    crop_times = crop_times

    with open(os.path.join(data_list_path, 'labelled_image.list'), 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]

    with open(os.path.join(data_list_path, 'labelled_label.list'), 'r') as f:
        label_list = f.readlines()
    label_list = [item.replace('\n', '') for item in label_list]

    for idx, image_name in enumerate(image_list):
        i_name = image_name.split('\\')[-1].split('.')[0]
        img_name = os.path.join(h5_save_path, i_name + "_")

        logger.info(f"第 {idx} 个数据：{i_name} 开始处理")
        st_time_1 = time.time()
        images, labels = read_data(data_path_img=image_list[idx], data_patch_lab=label_list[idx],
                                   is_rescale=True, norm_type=norm_type)
        st_time_2 = time.time()
        logger.info(f'读取数据所用时间为：{st_time_2 - st_time_1}s')
        covert_h5(images, labels, output_size=output_size_, fn=img_name, crop_times=crop_times, is_label=True)
        logger.info(f'转换数据所用时间为：{time.time() - st_time_2}s')

    logger.info(f' STS Labelled 二分类数据处理完毕。用时：{time.time() - st_time}s')


def sts_unlabelled_data(data_list_path, h5_save_path, crop_times=6, output_size=(112, 112, 80), norm_type=1):
    """
    处理二分类数据--Unlabelled：r''
    """
    st_time = time.time()
    make_dir(h5_save_path)

    logger.info(f'列表路径为：{data_list_path}')
    logger.info(f'保存路径为：{h5_save_path}')

    with open(data_list_path, 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]

    for idx, image_name in enumerate(image_list):
        i_name = image_name.split('\\')[-1].split('.')[0]
        img_name = os.path.join(h5_save_path, i_name + "-")

        logger.info(f"第 {idx} 个数据：{i_name} 开始处理")
        st_time_1 = time.time()

        images, _, _, _ = read_nii_image_data(data_path_img=image_list[idx], is_rescale=True, norm_type=norm_type)
        logger.info(f'images.type:{type(images)}, shape:{images.shape}')

        labels = np.zeros_like(images)

        st_time_2 = time.time()
        logger.info(f'读取数据所用时间为：{st_time_2 - st_time_1}s')

        covert_h5(images, labels, output_size=output_size, fn=img_name, crop_times=crop_times, is_label=False)
        logger.info(f'转换数据所用时间为：{time.time() - st_time_2}s')
        # break

    logger.info(f'数据处理完毕。用时：{time.time() - st_time}s')


def main():
    seed = 42
    np.random.seed(seed)

    crop_times = 20
    norm_type = 3
    output_size = (128, 128, 100)

    logger.debug(f'crop_times:{crop_times}, norm_type:{norm_type}, seed:{seed}, output_size:{output_size}')

    list_basic_path = r'..\dataset\nii_list'  # 列表路径
    un_list_1_path = os.path.join(list_basic_path, 'unlabelled_image.list')

    h5_basic_path = r'..\..\Data\STS-Data\rematch\sts_h5_data_crop_20'
    label_h5_path = os.path.join(h5_basic_path, 'labelled_image')
    un_h5_1_path = os.path.join(h5_basic_path, 'unlabelled_image')

    sts_label_data(data_list_path=list_basic_path, h5_save_path=label_h5_path,
                   output_size=output_size,
                   crop_times=crop_times, norm_type=norm_type)
    sts_unlabelled_data(data_list_path=un_list_1_path, h5_save_path=un_h5_1_path,
                        output_size=output_size,
                        crop_times=crop_times, norm_type=norm_type)


if __name__ == '__main__':
    main()
