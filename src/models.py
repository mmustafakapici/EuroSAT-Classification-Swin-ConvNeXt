import torch
import torch.nn as nn
import timm

def build_model(arch: str, num_classes: int, pretrained: bool = True, compile_model: bool = False):
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)  # PyTorch 2.x
    return model