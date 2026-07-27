"""Pattern-Aware ResNet50 - 95.69% Accuracy"""
import torch
import torch.nn as nn
from torchvision import models

class PatternAwareResNet(nn.Module):
    def __init__(self, num_classes=2, num_patterns=4):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_classify = nn.Linear(2048, num_classes)
        self.fc_patterns = nn.Linear(2048, num_patterns)
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        classification = self.fc_classify(features)
        patterns = torch.sigmoid(self.fc_patterns(features))
        return classification, patterns
