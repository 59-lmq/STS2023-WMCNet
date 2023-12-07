import os
from loguru import logger
from sklearn.model_selection import train_test_split
from utils.tools import make_dir

log_path = './log/02_split_data_list'
logger.add(log_path + '{time}.txt', rotation='00:00')


def main():
    """
    Using all train data to train the model.
    :return: None
    """
    h5_basic_path = r'..\..\Data\STS-Data\rematch\sts_h5_data_crop_20'
    list_basic_path = r"..\dataset\data_list\sts_h5_data_crop_20"

    train_all_list = []

    label_h5_path = os.path.join(h5_basic_path, 'labelled_image')
    label_h5_path = os.path.abspath(label_h5_path)
    label_full_path = [os.path.join(label_h5_path, file) for file in os.listdir(label_h5_path)]

    unlabelled_image_h5_path = os.path.join(h5_basic_path, 'unlabelled_image')
    unlabelled_image_h5_path = os.path.abspath(unlabelled_image_h5_path)
    unlabelled_image_full_path = [os.path.join(unlabelled_image_h5_path, file) for file in
                                  os.listdir(unlabelled_image_h5_path)]

    train_all_list.extend(label_full_path)
    train_all_list.extend(unlabelled_image_full_path)

    with open(os.path.join(list_basic_path, "train-all.list"), "w") as f:
        f.write('\n'.join(train_all_list))


def split_val_train_for_sup():
    """
    Split the train data into train and val data. The ratio is 8:2.
    Here we split for supervised learning.
    :return:
    """
    logger.info(f'开始 划分 H5 数据集')
    # 数据集的文件夹名称
    nii_image_path = r'..\..\Data\STS-Data\rematch\labelled\image'
    h5_image_path = r'..\..\Data\STS-Data\rematch\sts_h5_data_crop_20\labelled_image'

    train_list_path = r"..\dataset\data_list\sts_h5_data_crop_20"
    make_dir(train_list_path)

    names_list = os.listdir(nii_image_path)  # 获取数据名称
    names = [name.split('.')[0] for name in names_list]  # 去除后缀
    data_list = os.listdir(h5_image_path)  # 获取数据名称
    train_val_percentage = 0.2
    train_ids, val_ids = train_test_split(names, test_size=train_val_percentage, random_state=42)  # 随机划分
    logger.info(f'以 8:2 的占比划分为训练集、验证集')
    logger.info(f'训练集名称为：{train_ids}')
    logger.info(f'验证集名称为：{val_ids}')

    train_list, val_list = [], []

    for idx in data_list:
        # train-L-1_000_roi.h5
        idx_name = idx.split('_')[0]
        if idx_name in train_ids:
            train_list.append(idx)
        elif idx_name in val_ids:
            val_list.append(idx)
        else:
            logger.info(f'数据集中不存在 {idx}')

    # 路径位置自己改
    with open(os.path.join(train_list_path, 'train-sup.list'), 'w') as f:
        f.write('\n'.join(train_list))

    with open(os.path.join(train_list_path, 'val-sup.list'), 'w') as f:
        f.write('\n'.join(val_list))

    logger.info(f'数据集总数为：{len(names)}，训练集数量为：{len(train_list)}, 验证集数量为：{len(val_list)}, ')
    logger.info(f' H5 数据集二分类数据集划分完毕')


def split_val_train_for_semi():
    """
    Split the train data into train and val data. The ratio is 8:2.
    Here we split for semi-supervised learning.
    :return:
    """
    logger.info(f'开始 划分 H5 数据集')
    # 数据集的文件夹名称
    nii_image_path = r'..\..\Data\STS-Data\rematch\labelled\image'
    h5_image_path = r'..\..\Data\STS-Data\rematch\sts_h5_data_crop_20\labelled_image'
    un_h5_image_path = r'..\..\Data\STS-Data\rematch\sts_h5_data_crop_20\unlabelled_image'

    train_list_path = r"..\dataset\data_list\sts_h5_data_crop_20"
    make_dir(train_list_path)

    names_list = os.listdir(nii_image_path)  # 获取数据名称
    names = [name.split('.')[0] for name in names_list]  # 去除后缀
    data_list = os.listdir(h5_image_path)  # 获取数据名称
    train_val_percentage = 0.2
    train_ids, val_ids = train_test_split(names, test_size=train_val_percentage, random_state=42)  # 随机划分
    logger.info(f'以 8:2 的占比划分为训练集、验证集')
    logger.info(f'训练集名称为：{train_ids}')
    logger.info(f'验证集名称为：{val_ids}')

    train_list, val_list = [], []
    train_label_full_path = []
    train_unlabeled_full_path = []
    val_label_full_path = []

    for idx in data_list:
        # train-L-1_000_roi.h5
        idx_name = idx.split('_')[0]
        if idx_name in train_ids:
            train_list.append(idx)
            train_label_full_path.append(os.path.join(os.path.abspath(h5_image_path), idx))
        elif idx_name in val_ids:
            val_list.append(idx)
            val_label_full_path.append(os.path.join(os.path.abspath(h5_image_path), idx))
        else:
            logger.info(f'数据集中不存在 {idx}')

    for idx in os.listdir(un_h5_image_path):
        train_unlabeled_full_path.append(os.path.join(os.path.abspath(un_h5_image_path), idx))

    train_full_path = []
    train_full_path.extend(train_label_full_path)
    train_full_path.extend(train_unlabeled_full_path)

    # 路径位置自己改
    with open(os.path.join(train_list_path, 'train-semi.list'), 'w') as f:
        f.write('\n'.join(train_full_path))

    with open(os.path.join(train_list_path, 'val-semi.list'), 'w') as f:
        f.write('\n'.join(val_label_full_path))

    logger.info(f' 训练集数量为：{len(train_full_path)}, 验证集数量为：{len(val_label_full_path)}, ')
    logger.info(f' H5 数据集二分类数据集划分完毕')


if __name__ == '__main__':
    main()
    # split_val_train_for_sup()
    split_val_train_for_semi()
