from enum import Enum


class ModeloBase(Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    CONVNEXT = "convnext"
    INCEPTION_V3 = "inception_v3"
    VGG16 = "vgg16"