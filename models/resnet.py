import torch
from torch import nn

class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
             
    def forward(self, x):
        x = self.model(x)
        return x


class ResNet18(CustomResNet):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

class ResNet34(CustomResNet):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

class ResNet50(CustomResNet):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)

class ResNet101(CustomResNet):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)

class ResNet152(CustomResNet):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)


