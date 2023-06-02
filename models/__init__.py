from .resnet import (resnet8, resnet8x4, resnet8x4_double, resnet14, resnet20,
                     resnet32, resnet32x4, resnet44, resnet56, resnet110)
from .resnetv2 import (resnet18, resnet18x2, resnet34, resnet50, resnet34x4,
                       resnext50_32x4d, wide_resnet50_2)
from .shufflenetv2 import shufflenet_v2_x1_0 as ShuffleNetV2Imagenet
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .cifar_resnet import cifarresnet18, cifarresnet34

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'ResNet18': resnet18,
    'ResNet18Double': resnet18x2,
    'ResNet34': resnet34,
    'cifarresnet34': cifarresnet34,
    'cifarresnet18': cifarresnet18,
    'ResNet50': resnet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'ShuffleV2': ShuffleNetV2Imagenet,
}
