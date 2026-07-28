"""
train_models.py
---------------
Model factory helper providing get_model for classification models (NASNet, MobileNet, etc.)
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int = 3, pretrained: bool = False) -> nn.Module:
    model_name = model_name.lower()
    if "nasnet" in model_name:
        try:
            import timm
            model = timm.create_model("nasnetamobile", pretrained=pretrained, num_classes=num_classes)
        except Exception:
            # Fallback torchvision mobilenet or resnet if timm fails
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    elif "mobilenet" in model_name:
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    elif "resnet" in model_name:
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif "convnext" in model_name:
        model = models.convnext_tiny(pretrained=pretrained)
        model.classifier[2] = nn.Linear(768, num_classes)
        return model
    else:
        # Generic fallback
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
