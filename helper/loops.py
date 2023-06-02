from __future__ import division, print_function

import sys
import time

import torch
# from .distiller_zoo import DistillKL2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .util import AverageMeter, accuracy, reduce_tensor


def mixup_data(x, y, opt, alpha=0.4):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(opt.gpu)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix(data, targets, alpha=0.25):

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    # shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    # targets = (targets, shuffled_targets, lam)

    return data, None


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) 

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        
        input, target = batch_data
        
        data_time.update(time.time() - end)
        
        # input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================

        output = model(input)
        loss = opt.gamma * criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        image, target = data
        
        if opt.gpu is not None:
            image = image.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        if opt.use_aug == 'mixup':
            image, _, _, _ = mixup_data(image, target, opt)
        elif opt.use_aug == 'cutmix':
            image, _ = cutmix(image, target)

        # ===================forward=====================
        feat_s, logit_s = model_s(image, is_feat=True)

        
        feat_t, logit_t = model_t(image, is_feat=True)
        feat_t = [f.detach() for f in feat_t]

        temp = opt.kd_T

        # loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t, temp, opt)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_div 
        loss = opt.alpha * loss_div
        
        losses.update(loss.item(), image.size(0))

        metrics = accuracy(logit_s, target, topk=(1, 5))
        top1.update(metrics[0].item(), image.size(0))
        top5.update(metrics[1].item(), image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Loss {loss.avg:.4f}\t'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses
                ))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg, data_time.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    
    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_2 = AverageMeter()
    top5_2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader)

    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            
            input, target = batch_data

            if opt.gpu is not None:
                input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            feat, output = model(input, is_feat=True)

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            metrics = accuracy(output, target, topk=(1, 5))
            top1.update(metrics[0].item(), input.size(0))
            top5.update(metrics[1].item(), input.size(0))

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                        'GPU: {2}\t'
                        'Acc@1 {top1.avg:.3f}\t'
                        'Acc@5 {top5.avg:.3f}'.format(
                        idx, n_batch, opt.gpu,
                        top1=top1, top5=top5))
    
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret
    
    return [top1.avg, top5.avg, losses.avg]
