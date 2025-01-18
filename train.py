import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from sympy.physics.units import length
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

import os
import csv
from tqdm import tqdm

assert torch.__version__.split('.')[0] == '1', 'PyTorch版本错误，需要为1.x.x'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='SARDet', help='Dataset type, must be one of csv or coco or SARDet.')
    parser.add_argument('--coco_path', default='../Datasets/SARDet-100K', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'SARDet':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on SARDet-100K,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train_3000',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    # 在loss.csv文件中写入表头
    # TODO: 将每轮（Epoch）或每次迭代（Iteration）的loss写入文件，最好能直接输出曲线图（考虑使用tensorboard）。
    os.makedirs('./runs/train/', exist_ok=True)
    with open("./runs/train/loss.csv", 'w', newline='') as loss_file:
        loss_writer = csv.writer(loss_file)
        loss_writer.writerow(['Epoch', 'Iteration', 'cla_loss', 'reg_loss', 'running_loss'])

        # 建立modules文件夹存储训练结果
        os.makedirs('./runs/train/modules', exist_ok=True)   #创建目录
        for epoch_num in tqdm(range(parser.epochs)):
            retinanet.train()
            retinanet.module.freeze_bn()

            epoch_loss = []

            # 打开进度条
            iter_now = tqdm(total=len(dataloader_train))
            for iter_num, data in enumerate(dataloader_train):
                try:
                    optimizer.zero_grad()

                    if torch.cuda.is_available():
                        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                    else:
                        classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss = classification_loss + regression_loss

                    if bool(loss == 0):
                        continue

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                    optimizer.step()

                    loss_hist.append(float(loss))

                    epoch_loss.append(float(loss))

                    # print(
                    #     'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    #         epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                    loss_writer.writerow([epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)])
                    del classification_loss
                    del regression_loss
                    iter_now.update(1)  # 进度条更新
                except Exception as e:
                    print(e)
                    continue

            iter_now.close()    #关闭进度条

            if parser.dataset == 'coco' or parser.dataset == 'SARDet':

                print('Evaluating dataset')

                coco_eval.evaluate_coco(dataset_val, retinanet)

            elif parser.dataset == 'csv' and parser.csv_val is not None:

                print('Evaluating dataset')

                mAP = csv_eval.evaluate(dataset_val, retinanet)

            scheduler.step(np.mean(epoch_loss))

            torch.save(retinanet.module, './runs/train/modules/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()
    # TODO: 似乎没有判断最优模型的过程，直接存了最后一轮的权重参数？
    torch.save(retinanet, './runs/train/modules/model_final.pt')


if __name__ == '__main__':

    main()
