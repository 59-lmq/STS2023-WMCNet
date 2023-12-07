import os

import cv2
from loguru import logger

from utils.tools import load_h5

log_path = './log/03_visual_h5_data'
logger.add(log_path + '{time}.txt', rotation='00:00')


def load_and_show(h5_path):
    h5_list = os.listdir(h5_path)
    logger.info(f'h5_list:{h5_list[:5]}')

    for idx, file_name in enumerate(h5_list):
        file_path = os.path.join(h5_path, file_name)
        img, lab = load_h5(file_path)
        logger.info(f'{idx} ~ {file_name} ==> img:{img.shape}, lab:{lab.shape}')
        z_max = img.shape[-1]
        for z in range(z_max):
            show_img = img[:, :, z]
            show_lab = lab[:, :, z]
            cv2.imshow('show_img', show_img)
            cv2.imshow('show_lab', show_lab)
            cv2.waitKey(5)


def main():
    h5_path = r'..\..\Data\STS-Data\rematch\sts_h5_data_crop_20\labelled_image'
    load_and_show(h5_path)


if __name__ == '__main__':
    main()
