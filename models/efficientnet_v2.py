import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class CustomEfficientNetV2(nn.Module):
    def __init__(self, version, num_classes):
        super(CustomEfficientNetV2, self).__init__()
        self.model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', version, pretrained=True, nclass=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNetV2_s(CustomEfficientNetV2):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet_v2_s", num_classes=num_classes)

class EfficientNetV2_m(CustomEfficientNetV2):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet_v2_m", num_classes=num_classes)

class EfficientNetV2_l(CustomEfficientNetV2):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet_v2_l", num_classes=num_classes)
