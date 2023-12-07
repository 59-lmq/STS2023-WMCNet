from networks.vnet_cbam import VNet_CBAM
from networks.vnet import VNet
from networks.mcnet import MCNet3d_v1, MCNet3d_v2
from networks.wmcnet import WMCNet
from networks.wmcnet_no_cbam import WMCNet_NoCBAM

import torch
import torch.nn as nn


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def net_factory(net_type="vnet", in_chns=1, class_num=2, mode="train", init_weight="None"):
    """
    网络工厂
    :param net_type: 网络类型, vnet, VNet_CBAM, mcnet3d_v1, mcnet3d_v2, WMCNet, WMCNet_NoCBAM
    :param in_chns: 输入通道数
    :param class_num: 类别数
    :param mode: train or test
    :param init_weight: 初始化权重方式, kaiming_normal or xavier_normal or None
    :return: 网络
    """

    do_dropout = True if mode == "train" else False
    if net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=do_dropout)
    elif net_type == "VNet_CBAM":
        net = VNet_CBAM(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=do_dropout)
    elif net_type == "mcnet3d_v1":
        net = MCNet3d_v1(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=do_dropout)
    elif net_type == "mcnet3d_v2":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=do_dropout)
    elif net_type == "WMCNet":
        net = WMCNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=do_dropout)
    elif net_type == "WMCNet_NoCBAM":
        net = WMCNet_NoCBAM(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=do_dropout)

    if mode == 'train' and init_weight != "None":
        if init_weight == "kaiming_normal":
            net = kaiming_normal_init_weight(net)
        elif init_weight == "xavier_normal":
            net = xavier_normal_init_weight(net)

    net = net.cuda()

    return net
