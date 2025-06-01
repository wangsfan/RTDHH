import torch.nn as nn
from torchvision import models


# Todo add a relu_last_hidden=False parameter as in ResNetCifar
class ResNet(nn.Module):
    def __init__(self, num_classes=10, num_layers=18):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        if num_layers not in [18, 34, 50, 101, 152]:
            raise ValueError("For ResNet, choose a number of layers out of 18, 34, 50, 101, and 152")
        elif num_layers == 18:
            self.model = models.resnet18()
        elif num_layers == 34:
            self.model = models.resnet34()
        elif num_layers == 50:
            self.model = models.resnet50()
        elif num_layers == 101:
            self.model = models.resnet101()
        elif num_layers == 152:
            self.model = models.resnet152()

        self.latent_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()  # replace the builtin fully connected layer with an identity layer
        self.fc = nn.Linear(self.latent_dim, self.num_classes)
    def forward(self, input_):
        features = self.model(input_)
        output = self.fc(features)
        return output, features
