import math

import torch
import torch.nn.functional as F
import numpy as np


def single_case_mc(net, image, stride_xy, stride_z, ps, num_classes=1, is_average=False):
    """
    计算单个窗口的预测值，对单输入多输出的网络有效
    :param net: 网络
    :param image: 原图
    :param stride_xy: 滑动窗口沿 x y 轴的步长
    :param stride_z: 滑动窗口沿 z 轴的步长
    :param ps: patch-size，窗口大小
    :param num_classes: 类别数
    :param is_average: 是否求平均
    :return: label_map, score_map
    """
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < ps[0]:
        w_pad = ps[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < ps[1]:
        h_pad = ps[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < ps[2]:
        d_pad = ps[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image,
                       [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                       mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - ps[0]) / stride_xy) + 1
    sy = math.ceil((hh - ps[1]) / stride_xy) + 1
    sz = math.ceil((dd - ps[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - ps[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - ps[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - ps[2])
                test_patch = image[xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                if is_average:
                    y0 = net(test_patch)
                    y1 = torch.zeros(y0[0].shape).cuda()
                    for idx in range(len(y0)):
                        y1 += y0[idx]
                    y1 /= len(y0)
                    del y0
                else:
                    y1 = net(test_patch)[0]

                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]] += y
                cnt[xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]] += 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def single_case_vnet(net, image, stride_xy, stride_z, ps, num_classes=1):
    """
    计算单个窗口的预测值，对单输入单输出
    :param net: 网络
    :param image: 原图
    :param stride_xy: 滑动窗口沿 x y 轴的步长
    :param stride_z: 滑动窗口沿 z 轴的步长
    :param ps: patch-size，窗口大小
    :param num_classes: 类别数
    :return: label_map, score_map
    """
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < ps[0]:
        w_pad = ps[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < ps[1]:
        h_pad = ps[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < ps[2]:
        d_pad = ps[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image,
                       [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                       mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - ps[0]) / stride_xy) + 1
    sy = math.ceil((hh - ps[1]) / stride_xy) + 1
    sz = math.ceil((dd - ps[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - ps[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - ps[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - ps[2])
                test_patch = image[xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                y1 = net(test_patch)

                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]] += y
                cnt[xs:xs + ps[0], ys:ys + ps[1], zs:zs + ps[2]] += 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map
