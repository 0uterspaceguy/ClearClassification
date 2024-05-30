import torch
from torch import nn

class CustomMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
             
    def forward(self, x):
        x = self.model(x)
        return x

class MobileNetV2(CustomMobileNetV2):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(1280, num_classes)
                )