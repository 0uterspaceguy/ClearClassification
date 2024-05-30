import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class CustomEfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained(version, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNetB0(CustomEfficientNet):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet-b0", num_classes=num_classes)

class EfficientNetB1(CustomEfficientNet):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet-b1", num_classes=num_classes)

class EfficientNetB2(CustomEfficientNet):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet-b2", num_classes=num_classes)

class EfficientNetB3(CustomEfficientNet):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet-b3", num_classes=num_classes)

class EfficientNetB4(CustomEfficientNet):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet-b4", num_classes=num_classes)

class EfficientNetB5(CustomEfficientNet):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet-b5", num_classes=num_classes)

class EfficientNetB6(CustomEfficientNet):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet-b6", num_classes=num_classes)

class EfficientNetB7(CustomEfficientNet):
    def __init__(self, num_classes=2):
        super().__init__(version="efficientnet-b7", num_classes=num_classes)


