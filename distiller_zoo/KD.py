from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self):
        super(DistillKL, self).__init__()

    def forward(self, y_s, y_t, temp, opt):
        
        T= temp
        KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_s/T, dim=1),
                                    F.softmax(y_t/T, dim=1)) * T * T

        return KD_loss
