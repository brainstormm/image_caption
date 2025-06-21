import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.resnet(images)
        # (batch_size, 2048, 1, 1)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        # (batch_size, embed_size)
        features = self.bn(features)
        # (batch_size, embed_size)
        return features
