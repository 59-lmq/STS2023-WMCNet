import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.toothLoader import TwoTeeth, RandomCrop, RandomRotFlip, ToTensor
from networks.net_factory import net_factory
from utils.losses import dice_loss
from utils.ramps import get_lr
from utils.tools import make_dir
from utils.util import decode_seg_map_sequence
from utils.val_3d import val_all_case


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,
                        default='sup- WMCNet - CE+Dice-42',
                        help='训练名称')
    parser.add_argument('--net_name', type=str, default='WMCNet',
                        help='网络，vnet, VNet_CBAM, mcnet3d_v1, mcnet3d_v2, WMCNet, WMCNet_NoCBAM')
    parser.add_argument('--which_loss', type=int, default=1,
                        help='损失函数: 1-> CE+Dice')

    parser.add_argument("--data_list_path", type=str,
                        default=r'./dataset/data_list/sts_h5_data_crop_20',
                        help="data list path")
    parser.add_argument("--h5_image_path", type=str,
                        default=r'../Data/STS-Data/rematch/sts_h5_data_crop_20/labelled_image',
                        help="h5 image path")
    parser.add_argument("--val_image_path", type=str,
                        default=r'../Data/STS-Data/rematch/sts_val/image',
                        help="val image path")
    parser.add_argument("--val_label_path", type=str,
                        default=r'../Data/STS-Data/rematch/sts_val/label',
                        help="val label path")

    parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')

    parser.add_argument('--patch_size', type=tuple, default=(112, 112, 80), help='patch size per sample')

    parser.add_argument('--num_class', type=int, default=2, help='class of you want to segment')
    parser.add_argument('--base_lr', type=float, default=0.01, help='基础学习率')

    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate")
    parser.add_argument("--decay", type=float, default=0.0001, help="decay rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--optimizer", type=str, default="SGD",
                        help="optimization algorithm, Adam, AdamW or SGD")

    parser.add_argument('--lambda_dice', type=float, default=0.5, help='weight of dice loss')

    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
    parser.add_argument('--num_workers', type=int, default=4, help='num-workers to use')

    parser.add_argument('--use_val', type=bool, default=True, help='是否使用验证集')
    parser.add_argument('--val_iter', type=int, default=6000, help='每几轮验证一次')
    parser.add_argument('--early_stop_patience', type=int, default=7000, help='早停的 iter 间隔')

    args_ = parser.parse_args()

    return args_


def create_model(args):
    model = net_factory(net_type=args.net_name, in_chns=1, class_num=args.num_class, mode="train")
    return model


def main():
    main_st_time = time.time()
    args = get_args()
    snapshot_path = "../Experiments/STS2023-experiments/" + args.exp + "/"

    train_model_path = snapshot_path
    # train_model_path = snapshot_path + "ckp/"
    # make_dir(train_model_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size * len(args.gpu.split(','))
    max_iterations = args.max_iterations
    base_lr = args.base_lr

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    patch_size = args.patch_size

    # make logger file
    make_dir(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code')

    log_path = os.path.join(snapshot_path, 'log/train_')  # log_path: ../experiments/args.exp/log/train_xxx.txt
    logger.add(log_path + '{time}.txt', rotation='00:00')
    logger.info(args)

    model = create_model(args)
    model = model.cuda()

    data_path = {
        'train': os.path.join(args.data_list_path, 'train-sup.list'),
        'val': os.path.join(args.data_list_path, 'val-sup.list'),
    }

    db_train = TwoTeeth(data_path=data_path,
                        basic_path=args.h5_image_path,
                        split='train',
                        transform=transforms.Compose([
                            RandomCrop(patch_size),
                            RandomRotFlip(),
                            ToTensor(),
                        ]))

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.decay)
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    best_dice = 0
    best_iter = 0
    max_epoch = max_iterations // len(train_loader) + 1
    logger.info(f'共有{max_epoch}个epoch，每个epoch有{len(train_loader)}个iterations')

    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        loss_list = []

        for i_batch, sampled_batch in enumerate(train_loader):
            st_time = time.time()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            num_outputs = len(outputs)

            n, x, y, z = label_batch.size()
            one_hot_super = torch.zeros(n, args.num_class, x, y, z).cuda()
            one_hot_super = one_hot_super.scatter_(1, label_batch.view(n, 1, x, y, z), 1)

            ce_loss = 0
            loss_seg_dice = 0

            outputs_soft = 0

            for idx in range(num_outputs):
                y = outputs[idx]
                y_prob = F.softmax(y, dim=1)
                outputs_soft += y_prob

                ce_loss += F.cross_entropy(y, label_batch)
                loss_seg_dice += dice_loss(y_prob[:, 1], one_hot_super[:, 1])
            outputs_soft /= num_outputs

            loss = (1 - args.lambda_dice) * ce_loss + args.lambda_dice * loss_seg_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            writer.add_scalar('lr', get_lr(optimizer), iter_num)
            writer.add_scalar('loss/ce_loss', ce_loss, iter_num)
            writer.add_scalar('loss/dice_loss', loss_seg_dice, iter_num)
            writer.add_scalar('loss/total_loss', loss, iter_num)

            loss_list.append(loss.item())

            if iter_num > max_iterations:
                break

            if abs(best_iter - iter_num) > args.early_stop_patience:
                break

            if iter_num % 100 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                grid_image = make_grid(decode_seg_map_sequence(image), 5, normalize=False)
                writer.add_image('train/Prediction', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1).data.cpu().numpy()
                grid_image = make_grid(decode_seg_map_sequence(image), 5, normalize=False)
                writer.add_image('train/GroundTruth', grid_image, iter_num)

            logger.info(f'iter: {iter_num} / {args.max_iterations - 1}\t supervised loss : {loss.item():.5f}\t'
                        f'CE Loss:{ce_loss.item():.5f}\t Dice Loss:{loss_seg_dice.item():.5f}\t'
                        f'lr: {get_lr(optimizer)}\t time: {time.time() - st_time:.3f}s')

            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % args.val_iter == 0:
                if args.use_val:
                    model.eval()
                    metric_, metric_aug = val_all_case(model,
                                                       image_path=args.val_image_path,
                                                       label_path=args.val_label_path,
                                                       norm_type=3,
                                                       which_model=2,
                                                       num_classes=args.num_class,
                                                       patch_size=patch_size,
                                                       stride_xy=18, stride_z=4,
                                                       my_logger=logger)
                    val_dice, val_dice_aug = metric_['mean']['dice'], metric_aug['mean']['dice']
                    logger.info(f'{iter_num}次迭代的验证集dice为：{val_dice}, {val_dice_aug}')

                    if val_dice > best_dice:
                        best_dice = val_dice
                        best_iter = iter_num
                        dice_str = str(val_dice * 100).split('.')[0] + '_' + str(val_dice * 100).split('.')[1][0:2]

                        save_model_path = os.path.join(train_model_path, 'best_dice.pth')
                        save_epoch_path = os.path.join(train_model_path,
                                                       'iter_{}_dice_{}.pth'.format(iter_num, dice_str))

                        torch.save(model.state_dict(), save_model_path)
                        torch.save(model.state_dict(), save_epoch_path)

                        logger.info("save model to {}".format(save_model_path))
                    model.train()

                else:
                    save_model_path = os.path.join(train_model_path, 'iter_' + str(iter_num) + '.pth')
                    torch.save(model.state_dict(), save_model_path)
                    logger.info("save model to {}".format(save_model_path))

        writer.add_scalar('epoch_loss/loss', np.mean(loss_list), epoch_num)

        time2 = time.time()
        logger.info(f'One Train Epoch Used time: {time2 - time1}s')

        if iter_num > max_iterations:
            break

        if abs(best_iter - iter_num) > args.early_stop_patience:
            break

    save_model_path = os.path.join(train_model_path, 'final_epoch.pth')
    torch.save(model.state_dict(), save_model_path)
    logger.info("save model to {}".format(save_model_path))
    writer.close()
    logger.info(f'训练结束！共用时{time.time() - main_st_time}s，最佳dice为{best_dice}, iter为{best_iter}')


if __name__ == "__main__":
    main()
