
import os
from loguru import logger
from utils.tools import make_dir

log_path = './log/00_gain_data_list'
logger.add(log_path + '{time}.txt', rotation='00:00')


def nii_data_to_list():

    data_list_save_path = r'..\dataset\nii_list'
    make_dir(data_list_save_path)

    basic_path = r'..\..\Data\STS-Data\rematch'
    logger.info(f'basic_path: {basic_path} full_path: {os.path.abspath(basic_path)}')

    labelled_image_path = os.path.join(basic_path, 'labelled', 'image')
    labelled_label_path = os.path.join(basic_path, 'labelled', 'label')

    test_image_path = os.path.join(basic_path, 'test')

    unlabelled_image_path = os.path.join(basic_path, 'unlabelled_image')

    name_list = ['labelled_image', 'labelled_label', 'test_image', 'unlabelled_image']
    path_list = [labelled_image_path, labelled_label_path, test_image_path, unlabelled_image_path]

    for name, path in zip(name_list, path_list):
        file_list = os.listdir(path)
        file_list = [os.path.join(path, file) for file in file_list]
        with open(os.path.join(data_list_save_path, '{}.list'.format(name)), 'w') as f:
            f.write('\n'.join(file_list))


if __name__ == '__main__':
    nii_data_to_list()

