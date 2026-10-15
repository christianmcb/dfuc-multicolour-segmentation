"""Training loop, ported from the original train.py.

Logic, defaults, seed handling and logging are unchanged. Differences from the original,
none of which alters results on a GPU:
  * `.cuda()` calls became `.to(device)` so the code also runs on CPU (smoke test).
  * The colour transform is chosen by name (see colourspaces.py) instead of by importing a
    different Dataset class from one of five copies of this script.
  * Settings can come from a YAML file (`--config`); command-line flags override it.
  * Paths are relative to --data-root and --output-dir rather than hard-coded 'inputs/' and 'models/'.
  * `verbose=1` was dropped from ReduceLROnPlateau (it only printed LR changes and is deprecated).
"""
import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

from . import losses, models
from .colourspaces import COLOUR_TRANSFORMS, get_colour_transform
from .dataset import Dataset
from .metrics import iou_score, dice_coef
from .utils import AverageMeter, str2bool

ARCH_NAMES = models.__all__
LOSS_NAMES = list(losses.__all__)
LOSS_NAMES.append('BCEWithLogitsLoss')


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default=None,
                        help='YAML file of settings; command-line flags override it')
    parser.add_argument('--name', default=None,
                        help='model name: (default: dataset+arch)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' + ' | '.join(ARCH_NAMES) + ' (default: UNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=None, type=int,
                        help='input channels (default: set by --colour_transform)')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=640, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=480, type=int,
                        help='image height')

    # colour transform (new: replaces choosing a Dataset class by hand)
    parser.add_argument('--colour_transform', default='rgb', choices=sorted(COLOUR_TRANSFORMS),
                        help='channel set built from the BGR image (see colourspaces.py)')
    parser.add_argument('--luminance_power', default=5, type=int,
                        help='exponent for --colour_transform rgb_y (source file value: 5)')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dfuc2022',
                        help='dataset folder name under --data_root')
    parser.add_argument('--data_root', default='data',
                        help='folder containing <dataset>/{train,val}/{images,masks}')
    parser.add_argument('--output_dir', default='runs',
                        help='run folders (config.yml, log.csv, model.pth) are written here')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=6, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=20, type=int,
                        metavar='N', help='early stopping (default: 20; -1 disables)')

    parser.add_argument('--num_workers', default=4, type=int)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config is not None:
        with open(pre.config) as f:
            parser.set_defaults(**yaml.safe_load(f))
    config = parser.parse_args(argv)
    if config.input_channels is None:
        config.input_channels = COLOUR_TRANSFORMS[config.colour_transform][1]
    return config


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(config, train_loader, model, criterion, optimizer, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice_score': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice_score = dice_coef(output, target)

        # compute gradient and do optimising step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice_score'].update(dice_score, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice_score', avg_meters['dice_score'].avg)])


def validate(config, val_loader, model, criterion, device):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'val_dice_score': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                val_dice_score = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['val_dice_score'].update(val_dice_score, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('val_dice_score', avg_meters['val_dice_score'].avg)])


def make_criterion(config, device):
    if config['loss'] == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss().to(device)
    return losses.__dict__[config['loss']]().to(device)


def make_loader(config, split, colour_transform, shuffle, drop_last):
    img_dir = os.path.join(config['data_root'], config['dataset'], split, 'images')
    mask_dir = os.path.join(config['data_root'], config['dataset'], split, 'masks')
    img_ids = glob(os.path.join(img_dir, '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    print('%s_img_ids found: %d' % (split, len(img_ids)))
    dataset = Dataset(
        img_ids=img_ids,
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=None,
        colour_transform=colour_transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config['num_workers'],
        drop_last=drop_last)


def main(argv=None):
    config = vars(parse_args(argv))

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    run_dir = os.path.join(config['output_dir'], config['name'])
    os.makedirs(run_dir, exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(os.path.join(run_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    device = get_device()

    # define loss function (criterion)
    criterion = make_criterion(config, device)

    cudnn.benchmark = True

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # create model
    print("=> creating model %s" % config['arch'])

    model = models.UNet(num_classes=config['num_classes'],
                        input_channels=config['input_channels'])

    model = model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    kwargs = {'power': config['luminance_power']} if config['colour_transform'] == 'rgb_y' else {}
    colour_transform, n_channels = get_colour_transform(config['colour_transform'], **kwargs)
    if n_channels != config['input_channels']:
        raise ValueError('colour_transform %s gives %d channels but input_channels is %d'
                         % (config['colour_transform'], n_channels, config['input_channels']))

    train_loader = make_loader(config, 'train', colour_transform, shuffle=True, drop_last=True)
    val_loader = make_loader(config, 'val', colour_transform, shuffle=False, drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice_score', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice_score', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, device)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, device)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        # NB: as in the original, this records the initial LR, not the scheduled one.
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice_score'].append(train_log['dice_score'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice_score'].append(val_log['val_dice_score'])

        pd.DataFrame(log).to_csv(os.path.join(run_dir, 'log.csv'), index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(run_dir, 'model.pth'))
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
