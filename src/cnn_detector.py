"""
CNN-based detector using ResNet-50.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def build_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


def save_model(model: nn.Module, path: str) -> None:
    path_t = path if path.endswith(".pt") else f"{path}.pt"
    torch.save(model.state_dict(), path_t)


def load_model(path: str, device: Optional[str] = None) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50(num_classes=2, pretrained=False)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
