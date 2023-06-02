"""
the general training framework
"""

from __future__ import print_function

import argparse
import json
import logging
import math
import os
import re
import time

import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from dataset.cifar100 import get_cifar100_dataloaders
                              
from dataset.imagenet import imagenet_list, get_syn_train_imagenet_dataloader
from distiller_zoo import DistillKL
from helper.loops import train_distill as train
from helper.loops import validate
from helper.util import (adjust_learning_rate, parser_config_save,
                         reduce_tensor, save_dict_to_json,
                         set_logging_defaults)
from models import model_dict


split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    # basic
    parser.add_argument('--print-freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--experiments_dir', type=str, default='models',help='Directory name to save the model, log, config')
    parser.add_argument('--experiments_name', type=str, default='baseline')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'imagenette'], help='dataset')

    # model
    parser.add_argument('--model_t', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                'ResNet18', 'ResNet34', 'resnet8x4_double', 'ResNet18DualMask',
                                'cifarresnet18','cifarresnet34',
                                'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_50_2',
                                'vgg8', 'vgg11', 'vgg13', 'vgg11_imagenet', 'vgg16', 'vgg19', 'ResNet50',
                                'vgg16_bn', 'vgg11_bn', 'ShuffleV2'])

    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                'ResNet18', 'ResNet34', 'resnet8x4_double', 'ResNet18DualMask',
                                'cifarresnet18','cifarresnet34',
                                'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_50_2',
                                'vgg8', 'vgg11', 'vgg13', 'vgg11_imagenet', 'vgg16', 'vgg19', 'ResNet50',
                                'vgg16_bn', 'vgg11_bn', 'ShuffleV2'])

    parser.add_argument('--path-t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd'])

    parser.add_argument('-r', '--gamma', type=float, default=0.1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0.9, help='weight balance for KD')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    parser.add_argument('--save_model', action='store_true')

    parser.add_argument('--use_aug', type=str, default='none', choices=['none', 'mixup', 'cutmix'])
    parser.add_argument('--data_amount', type=str, default='400k', choices=['50k', '100k', '140k', '200k', '400k', '800k', '1280k', '1600k', '2000k'])
    parser.add_argument('--data_quality', type=str, default='4', choices=['1', '1.6', '1.8', '2', '3', '4'])
    parser.add_argument('--s_step', type=str, default='100', choices=['100', '150', '200'])
    
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8081', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--rank', default = 0)
    
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')

    # resume training
    parser.add_argument('--resume', action='store_true')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.02

    # set the path of model and tensorboard
    opt.model_path = './save/student_model'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name =  os.path.join(opt.experiments_dir, opt.experiments_name)

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    parser_config_save(opt, opt.save_folder)

    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    if segments[0] == 'resnext50':
        return segments[0] + '_' + segments[1]
    if segments[0] == 'vgg13' and segments[1] == 'imagenet':
        return segments[0] + '_' + segments[1]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')

    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    # TODO: reduce size of the teacher saved in train_teacher.py
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    
    print(model_t)

    if opt.dataset == 'cifar100':
        model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    elif opt.dataset == 'imagenet':
        checkpoint = torch.load(model_path, map_location=map_location)
        # new_state_dict = {}
        # for k,v in checkpoint['model'].items():
        #     new_state_dict[k[7:]] = v
        # model.load_state_dict(checkpoint['state'])
        model.load_state_dict(checkpoint)
    
    print('==> done')
    return model

total_time = time.time()
best_acc = 0

def main():
    opt = parse_option()
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(27)
        cudnn.deterministic = False
        cudnn.benchmark = True
        numpy.random.seed(27)

    class_num_map = {
        'cifar100': 100,
        'imagenet': 1000,
        'imagenette': 10,
    }
    if opt.dataset not in class_num_map:
        raise NotImplementedError(opt.dataset)
    n_cls = class_num_map[opt.dataset]

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    if opt.dataset == 'cifar100':
        module_args = {'num_classes': n_cls}
    else:
        module_args = {'num_classes': n_cls}
    model_s = model_dict[opt.model_s](**module_args)
    
    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)

    model_t.eval()
    model_s.eval()

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation

    module_list.append(model_t)
    
    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # dataloader
    if opt.dataset == 'cifar100':
        syn_train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,syn_data_amount=opt.data_amount, 
                                                                        syn_data_quality=opt.data_quality, syn_s_step=opt.s_step,
                                                                        num_workers=opt.num_workers)
    elif opt.dataset in imagenet_list:
        syn_train_loader, syn_train_sampler, val_loader = get_syn_train_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers, syn_data_amount=opt.data_amount, 
                                                                        syn_data_quality=opt.data_quality, syn_s_step=opt.s_step,
                                                                        multiprocessing_distributed=opt.multiprocessing_distributed)
    else:
        raise NotImplementedError(opt.dataset)


    best_acc = 0
    
    if opt.resume:
        path = os.path.join(opt.save_folder, '{}_latest.pth'.format(opt.model_s))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']+1
        model_s.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load previous model successful.')
    else:
        start_epoch = 1

    # routine
    for epoch in range(start_epoch, opt.epochs + 1):

        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            syn_train_sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        # Syn ImageNet
        train_acc, train_acc_top5, train_loss, data_time = train(epoch, syn_train_loader, module_list, criterion_list, optimizer, opt)
        
        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss, data_time]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss, data_time = reduced.tolist()

        test_acc, test_acc_top5, _ = validate(val_loader, model_s, criterion_cls, opt)

        best_model = False
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = True

            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            
            save_file = os.path.join(opt.save_folder, '{}_latest.pth'.format(opt.model_s))
            torch.save(state, save_file)

            test_merics = {
                            'test_acc': test_acc,
                            'test_acc_top5': test_acc_top5,
                            'best_acc': best_acc,
                            'epoch': epoch,
                            }

            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))


if __name__ == '__main__':
    main()
